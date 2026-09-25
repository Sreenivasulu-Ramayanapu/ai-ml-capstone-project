# Support Assistant — Zepto Policy RAG Service

A LangGraph-orchestrated Retrieval-Augmented Generation (RAG) service that answers
Zepto policy questions using ChromaDB retrieval, wrapped in a FastAPI `POST /ask`
endpoint. Ships with a graded **mock-LLM baseline** (`MOCK_LLM=1`, the default) and
an optional real-LLM extension (`MOCK_LLM=0`).

## Architecture

**Ingestion → Embedding → Retrieval → Generation**

1. **Ingestion** — [`support-assistant.py`](support-assistant.py)'s `_load_and_chunk_documents()` reads every
   `.txt` file under [`docs/`](docs/) (8 Zepto policy documents: delivery, returns,
   membership, tracking, cancellations, gift cards, support hours, etc.) and splits
   each into fixed-size 500-character chunks.
2. **Embedding** — `build_vector_store()` encodes every chunk with the
   `all-MiniLM-L6-v2` `SentenceTransformer` model and upserts the vectors, ids, and
   raw chunk text into the ChromaDB collection `document_chunks` (an in-memory
   `chromadb.Client()` collection, rebuilt once at process startup if empty).
3. **Retrieval** — the `retrieve_and_answer` LangGraph node embeds the incoming
   query with the same model and calls `collection.query(...)` to fetch the
   top-3 most similar chunks by cosine similarity. This step always runs for
   real in both `MOCK_LLM` modes — no API key or network call is required.
4. **Generation** — only the final answer-generation step branches on
   `MOCK_LLM`:
   - **Mock (`MOCK_LLM` unset or `1`, the graded baseline)**: `retrieve_and_answer`
     returns a canned string `f"Based on the retrieved context: {top_chunk_snippet}"`
     and `direct_answer` returns a fixed "I can only answer questions about Zepto
     policies right now." No LLM is called; the `AnswerSchema` (`answer`,
     `sources`, `confidence`) is populated deterministically in code.
   - **Optional (`MOCK_LLM=0`)**: `_llm_classify_intent`, `_llm_generate_grounded_answer`,
     and `_llm_generate_direct_answer` call a real LLM (via `openai`) using the
     role-context-task-format-length `PROMPT_TEMPLATE` (which includes a negative
     constraint and a few-shot example). Raw LLM output is validated against
     `AnswerSchema`; on a `ValidationError` it retries up to 2 additional times
     with a corrective instruction appended to the prompt before returning a
     clearly marked error response.

### LangGraph flow

```
            +-------------------+
  query --> |  classify_intent   |
            +-------------------+
                     |
      (conditional edge on intent)
              /              \
             v                v
  +----------------------+   +----------------+
  | retrieve_and_answer  |   | direct_answer  |
  | (policy_question)    |   | (general_q.)   |
  +----------------------+   +----------------+
             \                /
              v              v
                 END (AnswerSchema)
```

- `classify_intent` — keyword heuristic in mock mode (`delivery`, `return`,
  `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours` →
  `policy_question`, else `general_question`); LLM call in `MOCK_LLM=0` mode.
- `retrieve_and_answer` — always retrieves for real; answer generation branches
  on `MOCK_LLM`.
- `direct_answer` — canned string in mock mode; LLM call (no retrieval) in
  `MOCK_LLM=0` mode.

The conditional edge routing itself (`_route_on_intent`) does not depend on
`MOCK_LLM` — only the generation step inside each node does.

## Running locally

```powershell
cd support_assistant
pip install -r requirements.txt
Copy-Item support-assistant.py app.py
uvicorn app:app --reload
```

> Note: Python module names can't contain hyphens, so `uvicorn support-assistant:app`
> cannot import the file directly. Copy/rename it to `app.py` first (as above), or use
> the Docker image below, which performs this rename automatically during the build.

`MOCK_LLM` defaults to `1` (mock/graded baseline) — no API key needed.

## Example calls (MOCK_LLM left at default)

**1. Policy question (triggers retrieval):**

Request:
```json
POST /ask
{"query": "What is your return policy?"}
```

Response:
```json
{"answer":"Based on the retrieved context: doc_02 — Returns & Refunds: \"Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned with","sources":["2","10","5"],"confidence":1.0}
```

**2. General question (no retrieval):**

Request:
```json
POST /ask
{"query": "What is the capital of France?"}
```

Response:
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

*(Both responses were produced by directly invoking the compiled LangGraph graph
with these two queries; the FastAPI `/ask` endpoint returns the identical
`AnswerSchema` JSON shape.)*

## Docker

```powershell
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant
```

Then `POST http://localhost:7860/ask` with `{"query": "..."}`.

cd C:\Sreenivasulu\Ahold\ai-ml-capstone\ai-ml-capstone-project\support_assistant
Copy-Item .\support-assistant.py .\app.py -Force
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860 --reload

Invoke-RestMethod `
  -Uri http://127.0.0.1:7860/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is your return policy?"}'
