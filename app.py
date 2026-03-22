import chainlit as cl
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI, OpenAI
load_dotenv(override=True)  # loads variables from .env into the environment, override defaults

# Esta función se ejecuta cada vez que el usuario envía un mensaje
@cl.on_message
async def on_message(message: cl.Message):
    
    # -------------------------------------------------------------------------
    # 1) Azure AI Search (si está configurado vía variables de entorno)
    # -------------------------------------------------------------------------
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX")
    
    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    aoai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    docs_for_rag = []
    if endpoint and api_key and index_name:
        try:
            # 1. Pre-inicializar OpenAI para usar re-uso y generar embeddings
            llm = None
            vector_queries = None
            if aoai_endpoint and aoai_key:
                llm = AzureOpenAI(azure_endpoint=aoai_endpoint, api_key=aoai_key, api_version=aoai_version)
                # Generar embedding de la pregunta si hay modelo de embedding definido
                if aoai_embedding_deployment:
                    from azure.search.documents.models import VectorizedQuery
                    try:
                        emb_resp = llm.embeddings.create(input=[message.content], model=aoai_embedding_deployment)
                        vector_queries = [VectorizedQuery(vector=emb_resp.data[0].embedding, k_nearest_neighbors=3, fields="text_vector")]
                    except Exception as e:
                        print(f"Error generando embedding (pasando a búsqueda solo por texto): {e}")

            client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
            results = client.search(search_text=message.content, vector_queries=vector_queries, top=5)

            lines = ["Top results from Azure AI Search (top 5):\n"]
            for i, doc in enumerate(results):
                # Campos típicos: title, url/source, content/text
                title = doc.get("title") or doc.get("metadata_storage_name") or doc.get("id") or f"doc-{i+1}"
                url = doc.get("url") or doc.get("source") or doc.get("filepath")
                snippet = doc.get("content") or doc.get("chunk") or doc.get("chunk_text") or doc.get("text")
                if snippet:
                    s = str(snippet)
                    snippet = (s[:240] + "…") if len(s) > 240 else s
                part = f"{i+1}. {title}"
                if url:
                    part += f" — {url}"
                if snippet:
                    part += f"\n   {snippet}"
                lines.append(part)
                docs_for_rag.append({
                    "id": i + 1,
                    "title": title,
                    "url": url,
                    "snippet": snippet or ""
                })

            # If Azure OpenAI is configured, generate an answer grounded on these docs
            if aoai_endpoint and aoai_key and aoai_deployment:
                # Build a compact context with citations like [1], [2] …
                context_lines = []
                for d in docs_for_rag:
                    cite = f"[{d['id']}] {d['title']}" + (f" — {d['url']}" if d['url'] else "")
                    if d['snippet']:
                        cite += f"\n{d['snippet']}"
                    context_lines.append(cite)
                context = "\n\n".join(context_lines)

                system_prompt = (
                    "You are a helpful assistant. Answer the user using only the provided context. "
                    "Cite sources using square brackets like [1], [2]. If unsure, say you don't know."
                )
                user_prompt = (
                    f"User question: {message.content}\n\nContext:\n{context}"
                )

                # Guard against common misconfig: deployment accidentally set to API version
                if aoai_deployment and aoai_deployment[:4].isdigit() and aoai_deployment.count('-') >= 2:
                    await cl.Message(content=(
                        "Configuración inválida: `AZURE_OPENAI_DEPLOYMENT` parece ser una versión ("
                        f"{aoai_deployment}). Debe ser el NOMBRE EXACTO del deployment en AI Foundry,"
                        " por ejemplo `gpt-4o-mini` o el nombre que le diste."
                    )).send()
                    # Show search-only as fallback
                    await cl.Message(content="\n".join(lines)).send()
                    return

                try:
                    # Utiliza el SDK de Azure OpenAI para generar la respuesta
                    if not llm:
                        llm = AzureOpenAI(azure_endpoint=aoai_endpoint, api_key=aoai_key, api_version=aoai_version)
                        
                    resp = llm.chat.completions.create(
                        model=aoai_deployment,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                    )
                    answer = resp.choices[0].message.content if resp.choices else ""
                    if not answer:
                        answer = "No pude generar respuesta con el contexto disponible."

                    await cl.Message(content=answer).send()
                    # Also show sources list below the answer
                    await cl.Message(content="\n".join(lines)).send()
                    return
                except Exception as e:
                    await cl.Message(content=(
                        "❌ Azure OpenAI error. Posibles causas: nombre de deployment incorrecto, endpoint del proyecto/servicio mal configurado, "
                        f"o permisos. Detalle: {e}"
                    )).send()
                    # Fall back to showing search results only
                    await cl.Message(content="\n".join(lines)).send()
                    return

            # If OpenAI not configured, just show search results
            await cl.Message(content="\n".join(lines)).send()
            return
        except Exception as e:
            await cl.Message(content=f"❌ Azure Search error: {e}\nRevisa AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY y AZURE_SEARCH_INDEX.").send()
            # Si falla, continuamos con el demo PDF
    
    # -------------------------------------------------------------------------
    # 2) SIMULACIÓN PARA PROBAR LA INTERFAZ DIVIDIDA (fallback)
    # -------------------------------------------------------------------------
    pdf_path = "./documento_prueba.pdf"

    # Verificación de seguridad para la prueba
    if not os.path.exists(pdf_path):
        await cl.Message(
            content="⚠️ Falta el archivo. Por favor, coloca un archivo llamado 'documento_prueba.pdf' en esta carpeta."
        ).send()
        return

    # Definimos el elemento PDF de Chainlit. 
    # El atributo display="side" es lo que hace que se abra a la derecha.
    pdf_element = cl.Pdf(
        name="Contrato_Legal", 
        display="side", 
        path=pdf_path, 
        page=1 # Página que quieres que se abra por defecto
    )

    # REGLA DE ORO DE CHAINLIT: 
    # El texto de tu respuesta DEBE contener exactamente el 'name' que le diste 
    # al PDF ("Contrato_Legal") para que Chainlit lo convierta en un botón clickeable.
    respuesta_ia = (
        "Azure Search no está configurado todavía o ocurrió un error. "
        "Mientras tanto, aquí tienes el demo del PDF en modo lateral (Contrato_Legal).\n\n"
        "Para activar Azure Search, define AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY y AZURE_SEARCH_INDEX en tu .env."
    )

    # Enviamos la respuesta a la pantalla del usuario adjuntando el elemento
    await cl.Message(
        content=respuesta_ia,
        elements=[pdf_element]
    ).send()