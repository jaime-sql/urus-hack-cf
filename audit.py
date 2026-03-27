import json
import os
import re
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient


CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "audit-logs")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


def _get_blob_client(blob_name: str):
    service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)
    try:
        container.create_container()
    except Exception:
        pass
    return container.get_blob_client(blob_name)


def compute_grounding_score(answer: str, docs_used: list) -> float:
    """
    Score 0.0-1.0 que mide qué tan anclada está
    la respuesta en los documentos recuperados.
    """
    if not docs_used or not answer:
        return 0.0

    # Score 1: cuántos docs fueron citados por su nombre clave, ej: [NRP-23] o [LINEAMIENTOS_TECNICOS]
    total_docs = len(docs_used)
    cited_count = 0
    for d in docs_used:
        doc_id = str(d.get("id", ""))
        if doc_id and doc_id.lower() in answer.lower():
            cited_count += 1
    citation_score = cited_count / total_docs if total_docs > 0 else 0

    # Score 2: overlap de palabras entre respuesta y contexto recuperado
    # Filtramos stopwords comunes para no inflar el score con artículos/preposiciones
    stopwords = {"de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
                 "the", "of", "and", "to", "is", "in", "for", "that", "this", "with"}
    answer_words = set(answer.lower().split()) - stopwords
    context_words = set()
    for d in docs_used:
        context_words.update(d.get("snippet", "").lower().split())
    context_words -= stopwords

    overlap = len(answer_words & context_words)
    overlap_score = min(overlap / 15, 1.0)  # 15 palabras en común = score perfecto

    # Promedio ponderado: citas pesan más que overlap
    return round((citation_score * 0.6) + (overlap_score * 0.4), 2)


def log_interaction(
    user: str,
    question: str,
    answer: str,
    docs_used: list,
    response_time_ms: int,
    grounded: bool = True,
):
    grounding_score = compute_grounding_score(answer, docs_used)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "question": question,
        "answer_length": len(answer),
        "grounded": grounded,
        "grounding_score": grounding_score,
        "sources_used": [
            {"id": d["id"], "title": d["title"], "url": d.get("url", "")}
            for d in docs_used
        ],
        "response_time_ms": response_time_ms,
    }

    line = json.dumps(entry, ensure_ascii=False) + "\n"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    blob_name = f"audit_{today}.jsonl"

    if not CONNECTION_STRING:
        with open(blob_name, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"[audit] Guardado localmente en {blob_name}")
        return

    try:
        blob = _get_blob_client(blob_name)
        try:
            existing = blob.download_blob().readall().decode("utf-8")
        except Exception:
            existing = ""
        blob.upload_blob(existing + line, overwrite=True)
        print(f"[audit] Guardado en Blob Storage: {blob_name} | score: {grounding_score}")
    except Exception as e:
        print(f"[audit] Error guardando en Blob Storage: {e}")
        with open(blob_name, "a", encoding="utf-8") as f:
            f.write(line)