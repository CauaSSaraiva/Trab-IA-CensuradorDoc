Este repositório contém um projeto Python que utiliza o `uv` como gerenciador de pacotes e executor de comandos.

## Como começar

### Pré-requisitos

Certifique-se de ter o `uv` instalado em seu sistema. Se não tiver, você pode instalá-lo seguindo as instruções na [documentação oficial do uv](https://www.google.com/search?q=https://docs.astral.sh/uv/tutorial/installation/).

### Instalação

1.  Clone este repositório:
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd ia-m-uv
    ```
2.  Instale as dependências usando `uv`:
    ```bash
    uv sync
    ```


### Rodar o programa principal

Para executar o arquivo principal do projeto (`main.py`), utilize o comando `uv run`,
existem argumentos para serem utilizados junto da execução, o OBRIGATÓRIO é o caminho da imagem com --imagem
há outros argumentos opcionais, como o --gemini-key.

exemplo de execução:
```bash
uv run ia-m-uv --imagem .\5teste_documento.jpeg --ocr-idiomas pt --ocr-confianca 0.5
```



Este comando executa o módulo `ia-m-uv`, que, de acordo com a estrutura do projeto, provavelmente aponta para `src/ia_m_uv/main.py`.

## Estrutura do Projeto

A estrutura básica do projeto é a seguinte:

```
ia-m-uv/
├── src/
│   └── ia_m_uv/
│       ├── algoritmos/
│       │   └── gemini_integration.py
│       ├── __init__.py
│       ├── main.py
│       └── utils.py
└── README.md
```

- `src/ia_m_uv/main.py`: O ponto de entrada principal da aplicação.
- `src/ia_m_uv/algoritmos/gemini_integration.py`: Contém a lógica para integração com a API Gemini. https://aistudio.google.com/app/apikey
- `src/ia_m_uv/utils.py`: Módulo para funções utilitárias.
- `src/ia_m_uv/__init__.py`: Arquivo de inicialização do pacote Python.
