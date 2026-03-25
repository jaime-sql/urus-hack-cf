from time import time

import chainlit as cl
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI, OpenAI
from audit import log_interaction
from safety import check_text
load_dotenv(override=True)  # loads variables from .env into the environment, override defaults

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> cl.User | None:
    # Extraer primer nombre del email
    email = default_user.identifier
    first_name = email.split("@")[0].capitalize()
    
    # Sobreescribir el identifier con el primer nombre
    custom_user = cl.User(
        identifier=first_name,
        metadata={
            **default_user.metadata,
            "email": email,  # guardamos el email real en metadata
            "display_name": first_name
        }
    )
    return custom_user

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("attached_document_text", "")

    user = cl.user_session.get("user")
    if user:
        name = user.metadata.get("display_name", user.identifier)
        await cl.Message(
            content=f"Bienvenido, **{name}** 👋\n¿En qué puedo ayudarte hoy?"
        ).send()
        
# Esta función se ejecuta cada vez que el usuario envía un mensaje
@cl.on_message
async def on_message(message: cl.Message):
    msg_history = cl.user_session.get("message_history", [])
    pinned_docs = cl.user_session.get("attached_document_text", "")
    
    start_time = time()

    # -------------------------------------------------------------------------
    # 0) Procesar Contexto Dinámico (Archivos Adjuntos por el Usuario)
    # -------------------------------------------------------------------------
    if message.elements:
        for element in message.elements:
            if element.mime == "text/plain":
                with open(element.path, "r", encoding="utf-8") as f:
                    pinned_docs += f"\n--- INICIO DOCUMENTO ADJUNTO: {element.name} ---\n{f.read()}\n--- FIN DOCUMENTO ADJUNTO ---\n"
            elif element.mime == "application/pdf":
                try:
                    import PyPDF2
                    with open(element.path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        pdf_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                        pinned_docs += f"\n--- INICIO PDF ADJUNTO: {element.name} ---\n{pdf_text}\n--- FIN PDF ADJUNTO ---\n"
                except Exception as e:
                    await cl.Message(content=f"⚠️ No se pudo extraer el texto del PDF {element.name}: {e}").send()
        
        # Guardar en la sesión para recordarlo todo el chat
        cl.user_session.set("attached_document_text", pinned_docs)

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
    
    # Parámetros para optimizar respuestas y tokens
    search_top_k = int(os.getenv("SEARCH_TOP_K", "3"))
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    openai_top_p = float(os.getenv("OPENAI_TOP_P", "0.95"))
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    safety_result = check_text(message.content)
    if not safety_result["is_safe"]:
        blocked = ", ".join(safety_result["blocked_categories"])
        await cl.Message(
            content=f"⚠️ Tu mensaje no pudo ser procesado por contener contenido inapropiado ({blocked}). Por favor reformula tu pregunta."
        ).send()
        return
    
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
                        vector_queries = [VectorizedQuery(vector=emb_resp.data[0].embedding, k_nearest_neighbors=search_top_k, fields="text_vector")]
                    except Exception as e:
                        print(f"Error generando embedding (pasando a búsqueda solo por texto): {e}")

            client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
            results = client.search(search_text=message.content, vector_queries=vector_queries, top=search_top_k)

            lines = [f"Top results from Azure AI Search (top {search_top_k}):\n"]
            import re
            for i, doc in enumerate(results):
                # Campos típicos: title, url/source, content/text
                title = doc.get("title") or doc.get("metadata_storage_name") or f"Documento_Legal_{i+1}"
                # Limpiar el título para usarlo como 'Keyword' (quitar .pdf, .docx)
                clean_title = re.sub(r'\.[a-zA-Z0-9]+$', '', title)
                
                url = doc.get("url") or doc.get("source") or doc.get("filepath")
                raw_snippet = doc.get("content") or doc.get("chunk") or doc.get("chunk_text") or doc.get("text")
                full_snippet = str(raw_snippet) if raw_snippet else ""
                short_snippet = (full_snippet[:500] + "…") if len(full_snippet) > 500 else full_snippet

                part = f"- {clean_title}"
                if url:
                    part += f" — {url}"
                if short_snippet:
                    part += f"\n   {short_snippet}"
                lines.append(part)
                docs_for_rag.append({
                    "id": clean_title,  # Usamos el título limpio como identificador/palabra clave
                    "title": title,
                    "url": url,
                    "snippet": short_snippet,
                    "full_snippet": full_snippet
                })

            # If Azure OpenAI is configured, generate an answer grounded on these docs
            if aoai_endpoint and aoai_key and aoai_deployment:
                # Build a compact context with keyword citations instead of numbers
                context_lines = []
                for d in docs_for_rag:
                    cite = f"FUENTE DOCUMENTAL: [{d['id']}]" + (f" — Ubicación: {d['url']}" if d['url'] else "")
                    if d['snippet']:
                        cite += f"\nExtracto Legal:\n{d['snippet']}"
                    context_lines.append(cite)
                context = "\n\n".join(context_lines)

                system_prompt = (
                    "Eres un asistente experto legal y técnico. Tu objetivo es cruzar y evaluar documentos.\n"
                    "El usuario puede proveerte su propio 'Documento Adjunto' y hacerte una pregunta.\n"
                    "Utiliza PRINCIPALMENTE el contexto recuperado de la Base de Conocimiento (Azure AI Search) para conocer las reglas o leyes.\n"
                    "Si la información está disponible, responde detalladamente justificando tu análisis.\n"
                    "REGLA ESTRICTA DE CITACIÓN: Siempre que uses información del contexto legal, DEBES citar la fuente usando corchetes con el NOMBRE EXACTO de la FUENTE DOCUMENTAL provista, por ejemplo: [NRP-23_Normas] o [Ley_de_Bancos]. NUNCA uses números como [1]."
                )
                
                # Para evitar desbordar el contexto con historial repetido, inyectamos el RAG y los PDFs 
                # ÚNICAMENTE en el último mensaje actual del usuario.
                current_context = f"=== CONTEXTO LEGAL / REGLAS (Azure AI Search) ===\n{context}\n"
                if pinned_docs:
                    current_context += f"\n=== DOCUMENTOS DEL USUARIO A EVALUAR (Archivos Adjuntos) ===\n{pinned_docs}\n"

                current_msg = f"Pregunta o Instrucción del Usuario: {message.content}\n\n{current_context}"

                llm_messages = [{"role": "system", "content": system_prompt}]
                # Agregar el historial de la conversación (memoria limpia)
                llm_messages.extend(msg_history)
                # Agregar el mensaje actual con el contexto inyectado
                llm_messages.append({"role": "user", "content": current_msg})

                # --- INDICADOR DE CARGANDO / PENSANDO ---
                msg = cl.Message(content="Pensando ⏳")
                await msg.send()

                # Guard against common misconfig: deployment accidentally set to API version
                if aoai_deployment and aoai_deployment[:4].isdigit() and aoai_deployment.count('-') >= 2:
                    msg.content = (
                        "Configuración inválida: `AZURE_OPENAI_DEPLOYMENT` parece ser una versión ("
                        f"{aoai_deployment}). Debe ser el NOMBRE EXACTO del deployment en AI Foundry,"
                        " por ejemplo `gpt-4o-mini` o el nombre que le diste."
                    )
                    await msg.update()
                    await cl.Message(content="\n".join(lines)).send()
                    return

                try:
                    # Utiliza el SDK de Azure OpenAI para generar la respuesta
                    if not llm:
                        llm = AzureOpenAI(azure_endpoint=aoai_endpoint, api_key=aoai_key, api_version=aoai_version)
                        
                    resp = llm.chat.completions.create(
                        model=aoai_deployment,
                        messages=llm_messages,
                        temperature=openai_temperature,
                        top_p=openai_top_p,
                        max_tokens=openai_max_tokens,
                    )
                    answer = resp.choices[0].message.content if resp.choices else ""
                    if not answer:
                        answer = "No pude generar respuesta con el contexto disponible."

                    user = cl.user_session.get("user")
                    user_id = user.identifier if user else "anonymous"
                    elapsed_ms = int((time() - start_time) * 1000)
                    log_interaction(
                        user=user_id,
                        question=message.content,
                        answer=answer,
                        docs_used=docs_for_rag,
                        response_time_ms=elapsed_ms,
                        grounded=True,
                    )
                    
                    # Crear elementos de citación elegantes en Chainlit
                    text_elements = []
                    for d in docs_for_rag:
                        source_name = str(d['id'])
                        elem_content = f"**Documento:** {d['title']}\n"
                        if d['url']:
                            elem_content += f"**Ruta:** {d['url']}\n"
                        elem_content += f"\n---\n**Extracto Completo Recuperado:**\n{d['full_snippet']}"
                        
                        text_elements.append(
                            cl.Text(name=source_name, content=elem_content, display="side")
                        )

                    # Guardar en memoria la pregunta limpia de este turno y la respuesta para el futuro
                    msg_history.append({"role": "user", "content": message.content})
                    msg_history.append({"role": "assistant", "content": answer})
                    # Guardar solo los últimos 10 mensajes (5 turnos de ida y vuelta) para no saturar tokens
                    if len(msg_history) > 10:
                        msg_history = msg_history[-10:]
                    cl.user_session.set("message_history", msg_history)

                    # Si la respuesta no incluyó citas por formato, Chainlit al menos forzará botones al pie.
                    if pinned_docs and not docs_for_rag:
                        # Si el usuario solo subió un PDF y no obtuvo resultados de Azure
                        text_elements.append(
                            cl.Text(name="Documento Adjunto", content="Se evaluó el documento que subiste previamente.", display="side")
                        )

                    msg.content = answer
                    msg.elements = text_elements
                    await msg.update()
                    return
                except Exception as e:
                    msg.content = (
                        "❌ Azure OpenAI error. Posibles causas: nombre de deployment incorrecto, endpoint del proyecto/servicio mal configurado, "
                        f"o permisos. Detalle: {e}"
                    )
                    await msg.update()
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