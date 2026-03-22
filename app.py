import chainlit as cl
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Esta función se ejecuta cada vez que el usuario envía un mensaje
@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message by querying Azure AI Search if configured.

    Fallback: if Azure variables are missing, show a quick guidance message
    and keep the PDF demo so the UI remains functional.
    """

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX")

    if endpoint and api_key and index_name:
        try:
            client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
            results = client.search(message.content, top=5)

            lines = ["Top results from Azure AI Search (top 5):\n"]
            for i, doc in enumerate(results):
                # Be resilient to unknown field names
                title = doc.get("title") or doc.get("metadata_storage_name") or doc.get("id") or f"doc-{i+1}"
                url = doc.get("url") or doc.get("source") or doc.get("filepath")
                snippet = doc.get("content") or doc.get("chunk_text") or doc.get("text")
                if snippet:
                    snippet = (str(snippet)[:240] + "…") if len(str(snippet)) > 240 else str(snippet)
                part = f"{i+1}. {title}"
                if url:
                    part += f" — {url}"
                if snippet:
                    part += f"\n   {snippet}"
                lines.append(part)

            await cl.Message(content="\n".join(lines)).send()
            return
        except Exception as e:
            await cl.Message(content=f"❌ Azure Search error: {e}\n\nRevisa las variables `AZURE_SEARCH_*` y que el índice exista.").send()
            return

    # Fallback demo with side PDF if Azure Search isn’t configured yet
    pdf_path = "./documento_prueba.pdf"
    if os.path.exists(pdf_path):
        pdf_element = cl.Pdf(name="Contrato_Legal", display="side", path=pdf_path, page=1)
        await cl.Message(
            content=(
                "Azure Search no está configurado todavía. Define `AZURE_SEARCH_ENDPOINT`, "
                "`AZURE_SEARCH_API_KEY` y `AZURE_SEARCH_INDEX` para activarlo.\n\n"
                "Mientras tanto, aquí tienes el demo del PDF en modo lateral (Contrato_Legal)."
            ),
            elements=[pdf_element],
        ).send()
        return

    await cl.Message(
        content=(
            "Configura Azure AI Search con las variables de entorno `AZURE_SEARCH_ENDPOINT`, "
            "`AZURE_SEARCH_API_KEY` y `AZURE_SEARCH_INDEX`, o agrega `documento_prueba.pdf` para ver el demo lateral."
        )
    ).send()