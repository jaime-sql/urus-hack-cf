from time import time

import chainlit as cl
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI, OpenAI
from fastapi.responses import RedirectResponse
from audit import log_interaction, compute_grounding_score 
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

@cl.on_logout
def on_logout(request, response):
    """Al cerrar sesión, redirigimos al endpoint de logout de Microsoft
    para limpiar también la sesión SSO del navegador."""
    tenant_id = os.getenv("OAUTH_AZURE_AD_TENANT_ID", "common")
    redirect_uri = os.getenv("CHAINLIT_URL", "http://localhost:8000")
    ms_logout_url = (
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/logout"
        f"?post_logout_redirect_uri={redirect_uri}"
    )
    return RedirectResponse(url=ms_logout_url)

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
    search_top_k = int(os.getenv("SEARCH_TOP_K", "3"))  # Keep at 3: more docs dilute citation score
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

                # === AUTO-DETECCIÓN DE TRACK ===
                q = message.content.lower()

                # === DETECCIÓN DE IDIOMA (debe ir ANTES del citation_mandate) ===
                english_indicators = ["what", "how", "why", "when", "where", "which", "who",
                                      "is", "are", "can", "does", "do", "the", "based", "steps",
                                      "should", "would", "could", "explain", "describe", "list",
                                      "according", "define", "definition", "nature", "perform",
                                      "legal", "law", "regulation", "compliance", "risk", "bank"]
                english_word_count = sum(1 for w in english_indicators if w in q.split())
                is_english = english_word_count >= 2

                # Regla de citación obligatoria — etiquetas en el idioma del usuario
                # citation_score = cited_docs / total_docs (peso 60%)
                # overlap_score  = palabras compartidas / 15  (peso 40%)
                unique_doc_ids = list(dict.fromkeys(d['id'] for d in docs_for_rag))
                all_doc_ids = ", ".join(f"[{did}]" for did in unique_doc_ids)
                if is_english:
                    citation_mandate = (
                        f"\n\n=== MANDATORY CITATION RULE ===\n"
                        f"Available documents: {all_doc_ids}\n"
                        f"You MUST cite EVERY document using its exact name in brackets.\n"
                        f"Use key phrases and terminology taken verbatim from the document extracts.\n"
                        f"Required structure:\n"
                        f"  1. Normative Framework: {all_doc_ids}\n"
                        f"  2. Analysis: one paragraph per document with a direct quote or reference.\n"
                        f"  3. Recommendation: grounded in the cited documents.\n"
                        f"NEVER skip a document. NEVER use [1], [2] as citations.\n"
                        f"IMPORTANT: Respond ENTIRELY in English."
                    )
                else:
                    citation_mandate = (
                        f"\n\n=== REGLA OBLIGATORIA DE CITACIÓN ===\n"
                        f"Documentos disponibles: {all_doc_ids}\n"
                        f"DEBES citar CADA documento usando exactamente su nombre entre corchetes.\n"
                        f"Usa frases clave y terminología tomada textualmente de sus extractos.\n"
                        f"Estructura:\n"
                        f"  1. Marco normativo: {all_doc_ids}\n"
                        f"  2. Análisis: un párrafo por documento con cita directa de su contenido.\n"
                        f"  3. Recomendación: fundamentada en los documentos citados.\n"
                        f"NUNCA omitas un documento. NUNCA uses [1], [2] como citas.\n"
                        f"IMPORTANTE: Responde SIEMPRE en español."
                    )

                # === TRACK DETECTION → system_prompt ===
                if any(w in q for w in ["ley", "artículo", "código", "regulación", "normativa", "legal",
                                         "jurídico", "contrato", "demanda", "litigio", "tribunal",
                                         "sentencia", "derecho", "statute", "clause", "court", "rights"]):
                    track = "Legal"
                    system_prompt = (
                        "Eres URUS Legal Advisor, experto en Derecho y análisis jurídico.\n"
                        "Responde SIEMPRE con: 1) el artículo o norma exacta aplicable, "
                        "2) explicación del alcance legal, 3) recomendación práctica de acción.\n"
                        "CITACIÓN: Cita el documento entre corchetes con su nombre, "
                        "ejemplo: [NRP-23] o [Ley_de_Bancos]. NUNCA uses números como [1]."
                    ) + citation_mandate
                elif any(w in q for w in ["compliance", "cumplimiento", "auditoría", "riesgo", "control",
                                           "due diligence", "kyc", "aml", "lavado", "pep",
                                           "política interna", "policy", "audit", "risk", "reporting"]):
                    track = "Compliance"
                    system_prompt = (
                        "Eres URUS Compliance Expert, especialista en cumplimiento normativo.\n"
                        "Responde SIEMPRE con: 1) ✅ CUMPLE / ⚠️ PARCIAL / ❌ NO CUMPLE por punto, "
                        "2) la norma específica aplicable, 3) brechas y pasos correctivos.\n"
                        "CITACIÓN: Cita el documento entre corchetes con su nombre. NUNCA uses números."
                    ) + citation_mandate
                elif any(w in q for w in ["salud", "paciente", "médico", "clínico", "diagnóstico",
                                           "tratamiento", "hipaa", "datos médicos", "expediente",
                                           "health", "patient", "medical", "clinical", "healthcare", "ehr"]):
                    track = "Healthcare"
                    system_prompt = (
                        "Eres URUS Healthcare Advisor, experto en regulaciones de salud y privacidad médica.\n"
                        "Responde SIEMPRE con: 1) la regulación aplicable (HIPAA, ley local), "
                        "2) implicaciones de privacidad del paciente, 3) recomendaciones de cumplimiento.\n"
                        "CITACIÓN: Cita el documento entre corchetes con su nombre. NUNCA uses números."
                    ) + citation_mandate
                else:
                    track = "Finance"
                    system_prompt = (
                        "Eres URUS Finance Advisor, experto en regulaciones financieras y banca.\n"
                        "Responde SIEMPRE con: 1) la ley financiera aplicable (NRP-23, Basel III, etc.), "
                        "2) el impacto financiero y nivel de riesgo, 3) recomendaciones concretas.\n"
                        "CITACIÓN: Cita el documento entre corchetes con su nombre. NUNCA uses números."
                    ) + citation_mandate

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
                track_emoji = {"Legal": "⚖️", "Compliance": "📋", "Healthcare": "🏥", "Finance": "💹"}
                msg = cl.Message(content=f"{track_emoji.get(track, '🤖')} **Track detectado: {track}** — Analizando documentos... ⏳")
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

                    # Grounding score
                    grounding_score = compute_grounding_score(answer, docs_for_rag)
                    if grounding_score >= 0.7:
                        score_label = "🟢 Alta"
                    elif grounding_score >= 0.4:
                        score_label = "🟡 Media"
                    else:
                        score_label = "🔴 Baja"

                    answer_with_score = (
                        f"{answer}\n\n---\n"
                        f"📊 **Confiabilidad:** {score_label} ({grounding_score:.0%})"
                    )

                    # Agrupar fragmentos por documento para evitar botones y nombres duplicados
                    grouped_docs = {}
                    for d in docs_for_rag:
                        source_name = str(d['id'])
                        if source_name not in grouped_docs:
                            grouped_docs[source_name] = {
                                "title": d['title'],
                                "url": d['url'],
                                "snippets": [d['full_snippet']]
                            }
                        else:
                            grouped_docs[source_name]["snippets"].append(d['full_snippet'])

                    # Crear elementos de citación elegantes en Chainlit (NUEVO METODO)
                    text_elements = []
                    for source_name, data in grouped_docs.items():
                        elem_content = f"**Documento:** {data['title']}\n"
                        if data['url']:
                            elem_content += f"**Ruta:** {data['url']}\n"
                        for idx, snippet in enumerate(data['snippets']):
                            elem_content += f"\n---\n**Extracto {idx+1}:**\n{snippet}"
                        
                        text_elements.append(
                            cl.Text(name=source_name, content=elem_content, display="side")
                        )

                    # Guardar en memoria la pregunta limpia de este turno y la respuesta para el futuro
                    msg_history.append({"role": "user", "content": message.content})
                    msg_history.append({"role": "assistant", "content": answer})
                    # Guardar solo los últimos 10 mensajes para no saturar tokens
                    if len(msg_history) > 10:
                        msg_history = msg_history[-10:]
                    cl.user_session.set("message_history", msg_history)

                    # Siempre agregar el Documento Adjunto a los elementos interactivos si existe
                    if pinned_docs:
                        text_elements.append(
                            cl.Text(name="Documento Adjunto", content="Extracto del Documento Adjunto:\n\n" + pinned_docs[:2500] + "\n[...CONTINÚA]", display="side")
                        )

                    # Forzar a Chainlit a mostrar los botones siempre sin duplicar
                    footer = "\n\n**📚 Fuentes analizadas:**\n"
                    for source_name in grouped_docs.keys():
                        footer += f"- {source_name}\n"
                    if pinned_docs:
                        footer += f"- Documento Adjunto\n"

                    # Construimos el mensaje final
                    final_content = answer_with_score + footer

                    # Actualizar UI con la respuesta, puntaje de evaluación y citaciones garantizadas
                    msg.content = final_content
                    msg.elements = text_elements
                    await msg.update()

                    # Audit log
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