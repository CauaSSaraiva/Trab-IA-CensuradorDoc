import sys

from .utils import parse_args
from .algoritmos.text_extraction import EasyOCRExtractor
from .algoritmos.gemini_censor import gemini_censor_text

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
            
            if args.gemini_key:
                texto_bruto = extractor.extract_text_raw(
                    image_path=args.imagem,
                    confidence_threshold=args.ocr_confianca
                )
                # print(f"Texto bruto extraído da imagem:\n{texto_bruto}\n")
                resultado_interpretado = gemini_censor_text(texto_bruto, args.gemini_key)
                print("\nResultado interpretado pelo Gemini:\n")
                print(resultado_interpretado)
                texto_extraido = extractor.extract_text(
                    image_path=args.imagem,
                    confidence_threshold=args.ocr_confianca
                )
                print(f"\nTexto da imagem censurado pelo pacote:\n{texto_extraido}\n")
            else:
                texto_extraido = extractor.extract_text(
                    image_path=args.imagem,
                    confidence_threshold=args.ocr_confianca
                )
                print(f"\nTexto extraído da imagem:\n{texto_extraido}\n")
        
        return 0
    except Exception as e:
        print(f"Erro: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
