import re
import os
import cv2
import numpy as np

def is_sensitive(text: str) -> bool:
    """
    Verifica se o texto corresponde a padrões sensíveis (CPF, datas, placas, etc.)
    """
    patterns = [
        r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',  # CPF
        r'\b\d{2}/\d{2}/\d{4}\b',             # Data
        r'\b[A-Z]{3}-?\d{4}\b',               # Placa
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def censor_sensitive_data(results, image: np.ndarray, original_image_path: str = None):
    """
    Recebe resultados do EasyOCR e censura visualmente textos sensíveis na imagem.

    Args:
        results (list): Saída do EasyOCR: [(bbox, text, confidence), ...]
        image (np.ndarray): Imagem já carregada
        original_image_path (str): Caminho original da imagem, para nomear saída
        save (bool): Se True, salva imagem censurada

    Returns:
        list: Novos resultados, com textos sensíveis substituídos por '[CENSURADO]'
    """
    sanitized = []
    
    for (bbox, text, conf) in results:
        if is_sensitive(text):
            print(f"[!] Texto sensível detectado e censurado: {text}")

            # Desenhar retângulo preto
            pts = [tuple(map(int, point)) for point in bbox]
            top_left = min(pts, key=lambda p: p[0] + p[1])
            bottom_right = max(pts, key=lambda p: p[0] + p[1])
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

            # Substitui o texto no resultado
            sanitized.append((bbox, '[CENSURADO]', conf))
        else:
            sanitized.append((bbox, text, conf))

    # Salva a imagem censurada
    if original_image_path:
        output_dir = "censored_images"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(original_image_path)
        output_path = os.path.join(output_dir, f"censored_{filename}")
        cv2.imwrite(output_path, image)
        print(f"[✔] Imagem censurada salva em: {output_path}")

    return sanitized