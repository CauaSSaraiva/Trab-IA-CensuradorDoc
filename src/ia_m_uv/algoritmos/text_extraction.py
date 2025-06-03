"""
EasyOCR Text Extractor - Português Brasileiro, Inglês
Extrai texto de imagens usando EasyOCR (CNN + CRNN)
Funciona MUITO melhor para português que TrOCR!
"""

import easyocr
import cv2
import numpy as np
from PIL import Image
import os

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
                print("Imagem considerada RUIM — será pré-processada.")
                processed_image = self.preprocess_image(image_path)
            
            # EasyOCR processa a imagem e retorna lista de resultados
            # Cada resultado: ([coordenadas], texto, confiança)
            results = self.reader.readtext(processed_image)
            
            # Filtra por confiança e extrai apenas o texto
            filtered_texts = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    filtered_texts.append(text)
                    
            
            # Junta todos os textos encontrados
            final_text = ' '.join(filtered_texts)
            
            return final_text.strip()
            
        except Exception as e:
            return f"Erro ao processar imagem: {str(e)}"
    
    def extract_text_detailed(self, image_path):
        """
        Retorna informações detalhadas sobre o texto encontrado
        
        - debug ou quando precisar de coordenadas do texto (vai precisar)
        - Útil para criar interfaces que mostram onde o texto foi encontrado
        """
        try:
            results = self.reader.readtext(image_path)
            
            detailed_results = []
            for (bbox, text, confidence) in results:
                detailed_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,  # coordenadas dos cantos
                })
            
            return detailed_results
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def should_preprocess(self, img):
        reader = easyocr.Reader(['pt'], gpu=False)
        results = reader.readtext(img, detail=1, paragraph=False)

        confidences = [conf for (_, _, conf) in results]  

        media = sum(confidences) / len(confidences) if confidences else 0
        print(f"Confiança média: {media:.2f} (limiar: 0.6)")

        return media < 0.56  # Ajustado com testes
    
    def preprocess_image(self, image_path):
        """
        Pré-processamento opcional da imagem
        """
        
        # Carrega imagem
        # img = cv2.imread(image_path)
        
        # 1. Ler em escala de cinza
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


        blurred = cv2.GaussianBlur(img, (3, 3), 0) # ksize 3x3, sigmaY 0


        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



        # 4. Remover manchas pequenas (ruído)
        # O limiar ta baixo para não apagar letras digitais.

        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # Filtra por área. 
            if cv2.contourArea(c) < 13: # ir ajustando ? 
                cv2.drawContours(binarized, [c], -1, 255, -1) 

        
        
        # Extrai o nome base do arquivo da imagem original
        base_name = os.path.basename(image_path)
        output_filename = f"processed_{base_name}"

        output_directory = "processed_images"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_path = os.path.join(output_directory, output_filename)

        cv2.imwrite(output_path, binarized) 
        
        
        return binarized
    
    def extract_from_multiple_images(self, image_folder):
        """
        Processa todas as imagens de uma pasta
        
        ONDE ALTERAR:
        - Adicione filtros para tipos de arquivo específicos
        - Implemente processamento paralelo para muitas imagens
        """
        results = {}
        
        # Extensões de imagem suportadas
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_path = os.path.join(image_folder, filename)
                print(f"Processando: {filename}")
                results[filename] = self.extract_text(image_path)
        
        return results

def main():
    """
    Fluxo principal de teste do EasyOCRExtractor
    """
    image_path = "teste_documento.jpg"  # <- SUBSTITUA por sua imagem (nem to usando o main, e sim com pyton -m e os argumentos)
    

    extractor = EasyOCRExtractor(
        languages=['pt', 'en'],  # português + inglês para textos mistos
        use_gpu=False  # mude para True se tiver GPU NVIDIA (tenho amd)
    )
    

    print("\n" + "="*60)
    print("TEXTO EXTRAÍDO (confiança >= 0.5):")
    print("="*60)
    texto_conservador = extractor.extract_text(image_path, confidence_threshold=0.5)
    print(texto_conservador)
    
    print("\n" + "="*60)
    print("TEXTO EXTRAÍDO (confiança >= 0.3) - mais texto:")
    print("="*60)
    texto_liberal = extractor.extract_text(image_path, confidence_threshold=0.3)
    print(texto_liberal)
    
    # Informações detalhadas para debug
    print("\n" + "="*60)
    print("INFORMAÇÕES DETALHADAS:")
    print("="*60)
    detalhes = extractor.extract_text_detailed(image_path)
    for item in detalhes:
        if 'error' not in item:
            print(f"Texto: '{item['text']}' | Confiança: {item['confidence']:.2f}")


if __name__ == "__main__":
    main()

