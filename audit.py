import json
import os
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
        pass  # ya existe
    return container.get_blob_client(blob_name)


def log_interaction(
    user: str,
    question: str,
    answer: str,
    docs_used: list,
    response_time_ms: int,
    grounded: bool = True,
):
    """
    Guarda una entrada de auditoría en Azure Blob Storage.
    Cada día tiene su propio archivo .jsonl para facilitar consultas.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "question": question,
        "answer_length": len(answer),
        "grounded": grounded,
        "sources_used": [
            {"id": d["id"], "title": d["title"], "url": d.get("url", "")}
            for d in docs_used
        ],
        "response_time_ms": response_time_ms,
    }

    line = json.dumps(entry, ensure_ascii=False) + "\n"

    # Un archivo por día: audit_2026-03-25.jsonl
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    blob_name = f"audit_{today}.jsonl"

    if not CONNECTION_STRING:
        # Fallback local si no hay Blob Storage configurado
        with open(blob_name, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"[audit] Guardado localmente en {blob_name}")
        return

    try:
        blob = _get_blob_client(blob_name)

        # Si el blob ya existe, descarga el contenido y agrega la nueva línea
        try:
            existing = blob.download_blob().readall().decode("utf-8")
        except Exception:
            existing = ""

        blob.upload_blob(existing + line, overwrite=True)
        print(f"[audit] Entrada guardada en Blob Storage: {blob_name}")

    except Exception as e:
        print(f"[audit] Error guardando en Blob Storage: {e}")
        # Fallback local
        with open(blob_name, "a", encoding="utf-8") as f:
            f.write(line)