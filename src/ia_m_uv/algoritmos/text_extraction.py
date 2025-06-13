"""
EasyOCR Text Extractor - Português Brasileiro, Inglês
Extrai texto de imagens usando EasyOCR (CNN + CRNN)
Funciona MUITO melhor para português que TrOCR!
"""

import easyocr
import cv2
import numpy as np
# from PIL import Image
import os
from .text_censor import censor_sensitive_data

class EasyOCRExtractor:
    def __init__(self, languages=None, use_gpu=None):
        """
        Inicializa o extrator EasyOCR com valores padrão que podem ser sobrescritos
        
        Args:
            languages (list): Lista de idiomas ['pt', 'en']
            use_gpu (bool): Usar GPU se disponível (NVIDIA)
        
        """
        # Valores padrão
        self.languages = languages or ['pt', 'en']
        self.use_gpu = use_gpu if use_gpu is not None else False
        
        print("Carregando EasyOCR... (primeira vez demora ~30s)")
        
        self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        
        print(f"EasyOCR carregado! Idiomas: {languages}")
    
    def create_default_extractor():
        """Função fábrica que retorna um extrator com configurações padrão"""
        return EasyOCRExtractor()  # Usa os valores padrão definidos na classe
    
    def extract_text_raw(self, image_path=None, confidence_threshold=None):
        """
        Extrai texto bruto da imagem, sem censura.
        """
        confidence_threshold = confidence_threshold if confidence_threshold is not None else 0.5
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        results = self.reader.readtext(img)
        filtered_texts = []
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                filtered_texts.append(text)
        return ' '.join(filtered_texts).strip()
    
    def extract_text(self, image_path=None, confidence_threshold=None):
        """
        Extrai texto de uma imagem
        
        Args:
            image_path (str): Caminho para a imagem
            confidence_threshold (float): Confiança mínima (0.0 a 1.0)
            
        Returns:
            str: Texto extraído da imagem
            
        """
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else 0.5
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            # pré-processamento da imagem se detectar ruído:
            if not self.should_preprocess(img):
                print("Imagem considerada BOA — não será pré-processada.")
                processed_image = img
            else:
                print("Imagem considerada RUIM — será pré-processada se detectado alterações possíveis.")
                processed_image = self.preprocess_image(image_path)
            
            # EasyOCR processa a imagem e retorna lista de resultados
            # Cada resultado: ([coordenadas], texto, confiança)
            results = self.reader.readtext(processed_image)
            
            censurado = censor_sensitive_data(results, processed_image.copy(), image_path)
            
            # Filtra por confiança e extrai apenas o texto
            filtered_texts = []
            for (bbox, text, confidence) in censurado:
                if confidence >= confidence_threshold:
                    filtered_texts.append(text)
                    
            
            # Junta todos os textos encontrados
            final_text = ' '.join(filtered_texts)
            
            return final_text.strip()
            
        except Exception as e:
            return f"Erro ao processar imagem: {str(e)}"
    
    # def extract_text_detailed(self, image_path):
    #     """
    #     Retorna informações detalhadas sobre o texto encontrado
        
    #     - Útil para criar interfaces que mostram onde o texto foi encontrado
    #     """
    #     try:
    #         results = self.reader.readtext(image_path)
            
    #         detailed_results = []
    #         for (bbox, text, confidence) in results:
    #             detailed_results.append({
    #                 'text': text,
    #                 'confidence': confidence,
    #                 'bbox': bbox,  # coordenadas dos cantos
    #             })
            
    #         return detailed_results
            
    #     except Exception as e:
    #         return [{'error': str(e)}]
    
    def should_preprocess(self, img):
        reader = easyocr.Reader(['pt'], gpu=False)
        results = reader.readtext(img, detail=1, paragraph=False)

        confidences = [conf for (_, _, conf) in results]  

        media = sum(confidences) / len(confidences) if confidences else 0
        print(f"Confiança média: {media:.2f} (limiar: 0.56)")

        return media < 0.56  # Ajustado com testes
    
    def adjust_contrast_if_needed(self, img: np.ndarray) -> np.ndarray:
        """Aumenta o contraste se a variação for muito baixa (lavadassa)."""
        if np.std(img) < 20:
            print("detectada imagem lavada, aumentando contraste...")
            return cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        return img

    def denoise_if_noisy(self, img: np.ndarray) -> np.ndarray:
        """Aplica blur leve se a variação for muito alta."""
        if np.std(img) > 70:
            print("detectada imagem ruidosa, aplicando desfoque...")
            return cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def preprocess_image(self, image_path):
        """
        Pré-processamento adaptativo baseado na análise da imagem.
        Não aplica transformações destrutivas em imagens que já estão boas.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # possível ideia de corrigir se ela tiver rotada em uns 90° mas foi-se umas 3 horas tentando fazer isso funcionar sem falso positivo,
        # quem sabe um dia. Por enquanto fica só comentado ai pra voltar na ideia depois
        # img = self.correct_rotation(img)

        # por alguns testes, agora ele só aplica esses role quando algum deles de fato melhora o resultado do easyocr, mas é bom testar mais dps
        img = self.adjust_contrast_if_needed(img)
        img = self.denoise_if_noisy(img)

        
        output_directory = "processed_images"
        os.makedirs(output_directory, exist_ok=True)
        output_filename = f"processed_{os.path.basename(image_path)}"
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, img)
        
        return img
    
    # def extract_from_multiple_images(self, image_folder):
    #     """
    #     Processa todas as imagens de uma pasta
    #     """
    #     results = {}
        
    #     # Extensões de imagem suportadas
    #     extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    #     for filename in os.listdir(image_folder):
    #         if any(filename.lower().endswith(ext) for ext in extensions):
    #             image_path = os.path.join(image_folder, filename)
    #             print(f"Processando: {filename}")
    #             results[filename] = self.extract_text(image_path)
        
    #     return results



