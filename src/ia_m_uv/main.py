import sys

# from .algoritmos.gemini_integration import GeminiClient
from .utils import parse_args
from .algoritmos.text_extraction import EasyOCRExtractor

"""
  Exemplo de utilização:
  uv run ia-m-uv -p 'prompt aqui'
  
  O arquivo main.py será chamado, nele a utilização do 'utils' é vista, além de ser possível a partir dele chamar outros algoritmos, até mesmo passar algum argumento texto:
  
  -p OU --problema 'prompt vai aqui'
"""


def main() -> None:
    print("Meu projeto!")
    args = parse_args()

    try:
        # Se uma imagem foi fornecida, processe com OCR
        if args.imagem:
            print("\n--- Processando imagem com OCR ---")
            
            extractor = EasyOCRExtractor(
                languages=args.ocr_idiomas,
                use_gpu=args.ocr_gpu
            )
            
            texto_extraido = extractor.extract_text(
                image_path=args.imagem,
                confidence_threshold=args.ocr_confianca
          )
            
            print(f"Texto extraído da imagem:\n{texto_extraido}\n")
            
           
        
        return 0
    except Exception as e:
        print(f"Erro: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
