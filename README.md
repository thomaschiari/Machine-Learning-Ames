# Machine Learning: Ames

## Autores:
- [Eduardo Mendes Vaz](https://github.com/EduardoMVAz)
- [Thomas Chiari Ciocchetti de Souza](https://github.com/thomaschiari)

## Sobre o Projeto:

Este é um projeto de regressão utilizando o dataset [Ames](https://www.openintro.org/book/statdata/ames.csv), dividido em duas partes:

### Análise e Modelagem
Foi feito uma análise e um pré-processamento nos dados do dataset, afim de alcançar uma acurácia maior nas previsões. Vários testes foram feitos nos dados, alterações, análises, etc., e tudo pode ser encontrado em /notebooks/preprocessing.ipynb.

Sobre os testes de modelo e comparações, podem ser encontrados em /notebooks/modelling.ipynb.

### API
Uma API simples foi desenvolvida com FastAPI, e pode ser encontrada em /API. Mais instruções na seção "Como Usar".

## Referências e Ferramentas
Diversas ferramentas e Referências foram usadas para o desenvolvimento de todo o projeto.
Na parte de análises, utilizamos os próprios notebooks fornecidos juntamente com a proposta, e os seguintes projetos:

[House Pricing - Feature Engineering](https://www.kaggle.com/code/ayushmehra/house-price-feature-engineering#2.-Feature-Engineering)

[House Regression: Beginner Catboost](https://www.kaggle.com/code/jimmyyeung/house-regression-beginner-catboost-top-2)

Para auxílio com código e scripts, utilizamos as ferramentas GitHub Copilot e ChatGPT.

## Como Usar
Para utilizar o projeto, siga os seguintes passos:

Inicialize um ambiente virutal em python com o comando:

    Windows:
        python -m venv env

    Linux:
        python3 -m venv env

Ative seu ambiente virtual e execute o seguinte comando para instalar as dependências do projeto:

    pip install -r requirements.txt

Com as dependências instaladas, você pode explorar o projeto!

### Notebooks
A parte de notebooks é mais uma leitura, acesse os notebooks como instruído previamente para ler sobre como foram feitas as partes do projeto, seja a análise exploratória, o pré-processamento ou a modelagem.

### API
A API pode ser executada com o seguinte comando:

    uvicorn main:app --reload

Assim que o comando for executado, a API será inicializada, e pode ser acessada na porta 8000, no caminho /predict (http://localhost:8000/predict/).

Utilize um programa para testar a API, como o Postman por exemplo.

A API recebe um corpo em formato JSON que representa uma linha do dataframe Ames, e utiliza nosso melhor modelo treinado para retornar uma previsão.

**OBSERVAÇÃO**: Está API não é robusta o suficiente para tratar erros. A intenção de seu desenvolvimento é permitir que o usuário inpute dados no modelo apresentado no Ames e receba uma previsão rápida do target. Para utilizá-la de forma correta, faça questão de acessar a documentação do dataset, e enviar os dados para a API de forma condizente.