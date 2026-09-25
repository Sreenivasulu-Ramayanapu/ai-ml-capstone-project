FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# uvicorn needs a valid module name (no hyphens), so copy the source to app.py
COPY support-assistant.py app.py
COPY docs/ docs/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
