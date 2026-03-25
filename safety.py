import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")

# Umbral de severidad 0-6 (0=safe, 6=muy peligroso)
# Para entorno corporativo, 2 es suficientemente estricto
THRESHOLD = int(os.getenv("CONTENT_SAFETY_THRESHOLD", "2"))


def check_text(text: str) -> dict:
    """
    Analiza un texto y retorna:
    - is_safe: True si el texto es seguro
    - blocked_categories: lista de categorías que superaron el umbral
    - scores: scores de cada categoría
    """
    if not ENDPOINT or not KEY:
        # Si no está configurado, pasa todo como seguro
        return {"is_safe": True, "blocked_categories": [], "scores": {}}

    try:
        client = ContentSafetyClient(ENDPOINT, AzureKeyCredential(KEY))

        request = AnalyzeTextOptions(
            text=text[:1000],  # límite de caracteres
            categories=[
                TextCategory.HATE,
                TextCategory.SELF_HARM,
                TextCategory.SEXUAL,
                TextCategory.VIOLENCE,
            ],
        )

        response = client.analyze_text(request)

        scores = {
            "hate": response.categories_analysis[0].severity,
            "self_harm": response.categories_analysis[1].severity,
            "sexual": response.categories_analysis[2].severity,
            "violence": response.categories_analysis[3].severity,
        }

        blocked = [cat for cat, score in scores.items() if score > THRESHOLD]

        return {
            "is_safe": len(blocked) == 0,
            "blocked_categories": blocked,
            "scores": scores,
        }

    except Exception as e:
        print(f"[safety] Error en Content Safety check: {e}")
        # En caso de error, permitir el paso (fail open)
        return {"is_safe": True, "blocked_categories": [], "scores": {}}