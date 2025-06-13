# algoritmos/gemini_censor.py
from typing import Dict, Optional
from .gemini_integration import GeminiClient  

DEFAULT_INSTRUCTION = (
    "Você é um assistente que detecta linguagem inadequada, ofensiva, sugestiva, sensível "
    "ou que contenha dados pessoais (como CPF, RG, datas de nascimento, números de documentos, "
    "endereços, nomes completos, etc) em textos extraídos de imagens. "
    "Se houver qualquer conteúdo desse tipo, aponte o trecho e explique brevemente. "
    "Caso não haja, responda apenas com 'OK'."
)

def gemini_censor_text(
    text: str,
    api_key: Optional[str] = None,
    instruction: Optional[str] = DEFAULT_INSTRUCTION,
    model_name: str = "gemini-1.5-flash"
) -> Dict:
    """
    Usa o Gemini para avaliar e censurar texto de forma mais contextual.

    Args:
        text (str): Texto extraído da imagem (já limpo pelo OCR).
        api_key (str): Chave da API Gemini (ou usa a variável de ambiente).
        instruction (str): Instrução para o modelo.
        model_name (str): Modelo a ser usado.

    Returns:
        dict: {
            "censored": bool,
            "reason": str (explicação ou 'OK'),
            "rephrased": Optional[str]
        }
    """
    client = GeminiClient(api_key=api_key, model_name=model_name)
    prompt = text.strip()

    response = client.generate_response_instructed(
        prompt=prompt,
        instruction=instruction
    )

    # Heurística simples: se resposta for só "OK", está limpo
    if response.strip().upper() == "OK":
        return {
            "censored": False,
            "reason": "Texto considerado aceitável pelo Gemini.",
            "rephrased": None
        }

    # Tenta gerar reescrita se desejado
    rephrase_response = client.generate_response_instructed(
        prompt=f"Reescreva o seguinte texto de forma segura, neutra e sem conteúdo sensível:\n\n{prompt}",
        instruction="Você é um filtro de segurança de conteúdo. Reescreva o texto de forma neutra."
    )

    return {
        "censored": True,
        "reason": response.strip(),
        "rephrased": rephrase_response.strip() if rephrase_response else None
    }
