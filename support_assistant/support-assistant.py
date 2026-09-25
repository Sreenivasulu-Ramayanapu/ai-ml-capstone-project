import glob
import contextlib
import io
import os
from typing import List, Literal, Optional, TypedDict

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END

# MOCK_LLM unset or "1" -> mock/graded-baseline branch (no LLM calls).
# MOCK_LLM="0" -> optional real-LLM extension branch.
MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

KEYWORD_TRIGGERS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 500
TOP_K = 3

# ---------------------------------------------------------------------------
# Ingestion + embedding: load every doc under docs/, chunk it, embed each
# chunk with all-MiniLM-L6-v2, and store the vectors in a ChromaDB collection.
# ---------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.Client()
collection = _chroma_client.get_or_create_collection("document_chunks")


def _load_and_chunk_documents() -> List[str]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for i in range(0, len(text), CHUNK_SIZE):
            chunks.append(text[i:i + CHUNK_SIZE])
    return chunks


def build_vector_store() -> None:
    if collection.count() > 0:
        return
    chunks = _load_and_chunk_documents()
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
    )


build_vector_store()

# Role-context-task-format-length prompt template with a negative constraint and a few-shot example.
PROMPT_TEMPLATE = """
Role: Support Assistant
Context: {context}
Task: Answer the user's question using only the information present in the context above.
Format: Respond with a short, direct answer.
Length: Keep the answer to 1-3 sentences.
Negative Constraint: Do not answer using information not present in the provided context.

Example:
Context: "'Pride and Prejudice' is a novel written by Jane Austen."
User: Who wrote 'Pride and Prejudice'?
Assistant: Jane Austen

User: {query}
Assistant:
"""


# ---------------------------------------------------------------------------
# Graded output schema
# ---------------------------------------------------------------------------
class AnswerSchema(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class AskRequest(BaseModel):
    query: str



# ---------------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    intent: Optional[Literal["policy_question", "general_question"]]
    retrieved_chunks: List[dict]
    result: Optional[AnswerSchema]


def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()
    if MOCK_LLM:
        intent = "policy_question" if any(kw in query_lower for kw in KEYWORD_TRIGGERS) else "general_question"
    else:
        intent = _llm_classify_intent(state["query"])
    return {**state, "intent": intent}


def _route_on_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def retrieve_and_answer(state: GraphState) -> GraphState:
    query_embedding = embedding_model.encode([state["query"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    ids = results["ids"][0]
    docs = results["documents"][0]
    retrieved_chunks = [{"id": chunk_id, "text": text} for chunk_id, text in zip(ids, docs)]

    if MOCK_LLM:
        top_chunk_snippet = retrieved_chunks[0]["text"][:200] if retrieved_chunks else ""
        answer = AnswerSchema(
            answer=f"Based on the retrieved context: {top_chunk_snippet}",
            sources=ids,
            confidence=1.0,
        )
    else:
        answer = _llm_generate_grounded_answer(state["query"], retrieved_chunks, ids)

    return {**state, "retrieved_chunks": retrieved_chunks, "result": answer}


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = AnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        answer = _llm_generate_direct_answer(state["query"])
    return {**state, "retrieved_chunks": [], "result": answer}


# ---------------------------------------------------------------------------
# Optional MOCK_LLM=0 extension: real LLM calls with schema-validation retries.
# ---------------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _validate_with_retries(build_prompt_fn, sources: List[str], max_retries: int = 2) -> AnswerSchema:
    prompt = build_prompt_fn(corrective_instruction="")
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw = _call_llm(prompt)
        try:
            return AnswerSchema.model_validate({
                "answer": raw.strip(),
                "sources": sources,
                "confidence": 0.8,
            })
        except ValidationError as exc:
            last_error = exc
            prompt = build_prompt_fn(
                corrective_instruction=f"Your previous output was invalid ({exc}). "
                "Return a plain text answer only, no JSON, no extra formatting."
            )
    return AnswerSchema(
        answer=f"Error: LLM output failed schema validation after {max_retries + 1} attempts ({last_error}).",
        sources=sources,
        confidence=0.0,
    )


def _llm_classify_intent(query: str) -> str:
    prompt = (
        "Classify the following user query as exactly one word: "
        "'policy_question' if it needs retrieval from a Zepto policy document, "
        "or 'general_question' otherwise.\nQuery: " + query
    )
    raw = _call_llm(prompt).strip().lower()
    return "policy_question" if "policy" in raw else "general_question"


def _llm_generate_grounded_answer(query: str, retrieved_chunks: List[dict], ids: List[str]) -> AnswerSchema:
    context = "\n".join(chunk["text"] for chunk in retrieved_chunks)

    def build_prompt(corrective_instruction: str) -> str:
        base = PROMPT_TEMPLATE.format(context=context, query=query)
        return base + ("\n" + corrective_instruction if corrective_instruction else "")

    return _validate_with_retries(build_prompt, sources=ids)


def _llm_generate_direct_answer(query: str) -> AnswerSchema:
    def build_prompt(corrective_instruction: str) -> str:
        base = f"Answer the following question concisely: {query}"
        return base + ("\n" + corrective_instruction if corrective_instruction else "")

    return _validate_with_retries(build_prompt, sources=[])


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)

graph_builder.set_entry_point("classify_intent")
graph_builder.add_conditional_edges(
    "classify_intent",
    _route_on_intent,
    {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
)
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AnswerSchema)
def ask(request: AskRequest) -> AnswerSchema:
    final_state = graph.invoke({
        "query": request.query,
        "intent": None,
        "retrieved_chunks": [],
        "result": None,
    })
    return final_state["result"]