# Utilitários do pacote
import argparse

"""
    parse_args
    Função para parsear os argumentos da linha de comando  
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Meu pacote python xyz")

    # parser.add_argument(
    #     "-p",
    #     "--problema",
    #     required=True,
    #     help="Problema exemplo a ser resolvido",
    # )
    # parser.add_argument(
    #     'imagem', 
    #     help='Caminho para a imagem a ser processada'
    # )
    # parser.add_argument(
    #     '--confianca', 
    #     type=float, 
    #     default=0.5, 
    #     help='Limite de confiança para OCR (0.0 a 1.0)'
    # )
    # parser.add_argument(
    #     '--gpu', 
    #     action='store_true', 
    #     help='Usar GPU para OCR se disponível', 
    #     default=False,
    #     nargs='?'
    # )
        # Argumentos do OCR (EasyExtractor)
    parser.add_argument('--imagem', required=True, help="Caminho da imagem para OCR")
    parser.add_argument('--ocr-idiomas', nargs='+', default=['pt', 'en'], 
                       help="Idiomas para o OCR (ex: 'pt en')")
    parser.add_argument('--ocr-gpu', action='store_true', 
                       help="Usar GPU para o OCR (se disponível)")
    parser.add_argument('--ocr-confianca', type=float, default=0.7,
                       help="Limite de confiança do OCR (0.0 a 1.0)")

    # Argumentos do Gemini (agora o usuario escolhe o token)
    # parser.add_argument('--gemini-token', required=True,
    #                    help="API Key do Gemini")

    return parser.parse_args()
