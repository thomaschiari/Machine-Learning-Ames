# Machine Learning: Ames

## Autores:
- [Eduardo Mendes Vaz](https://github.com/EduardoMVAz)
- [Thomas Chiari Ciocchetti de Souza](https://github.com/thomaschiari)

## Sobre o Projeto:

Este é um projeto de regressão utilizando o dataset [Ames](https://www.openintro.org/book/statdata/ames.csv), dividido em duas partes:

### Análise e Modelagem
Foi feito uma análise e um pré-processamento nos dados do dataset, afim de alcançar uma acurácia maior nas previsões. Vários testes foram feitos nos dados, alterações, análises, etc., e tudo pode ser encontrado em /notebooks/preprocessing.ipynb.

Sobre os testes de modelo e comparações, podem ser encontrados em /notebooks/modelling.ipynb. Lá, também é possível encontrar nossas conclusões, algumas explicações sobre os modelos escolhidos para a API e análise de importância de features em alguns modelos.

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

Para realizar a previsão, realize um POST request utilizando o URL:
    
    http://localhost:8000/predict/{model_name}
    
Com `model_name` sendo:
- `catboost` para CatBoost Regression;
- `ridge` para Regressão Linear com regularização Ridge;
- `both` para ambos os modelos, com uma média entre as previsões.

A API recebe um corpo em formato JSON que representa uma linha do dataframe Ames, e utiliza nosso melhor modelo treinado para retornar uma previsão. Dentro do notebook [`preprocessing`](./notebooks/preprocessing.ipynb), a última célula seleciona uma linha aleatória do dataset, e a transforma em um JSON, que pode ser utilizado para testar a API.

**OBSERVAÇÃO**: Está API não é robusta o suficiente para tratar erros. A intenção de seu desenvolvimento é permitir que o usuário inpute dados no modelo apresentado no Ames e receba uma previsão rápida do target. Para utilizá-la de forma correta, faça questão de acessar a documentação do dataset, e enviar os dados para a API de forma condizente.

Exemplo de inserção de dados:

```json

{
    "Order": 2640,
    "PID": 902104020,
    "MS.SubClass": 50,
    "MS.Zoning": "RM",
    "Lot.Frontage": 60.0,
    "Lot.Area": 9600,
    "Street": "Pave",
    "Alley": "Grvl",
    "Lot.Shape": "Reg",
    "Land.Contour": "Lvl",
    "Utilities": "AllPub",
    "Lot.Config": "Inside",
    "Land.Slope": "Gtl",
    "Neighborhood": "OldTown",
    "Condition.1": "Norm",
    "Condition.2": "Norm",
    "Bldg.Type": "1Fam",
    "House.Style": "1.5Fin",
    "Overall.Qual": 6,
    "Overall.Cond": 8,
    "Year.Built": 1900,
    "Year.Remod.Add": 2004,
    "Roof.Style": "Gable",
    "Roof.Matl": "CompShg",
    "Exterior.1st": "Wd Sdng",
    "Exterior.2nd": "Wd Sdng",
    "Mas.Vnr.Type": null,
    "Mas.Vnr.Area": 0.0,
    "Exter.Qual": "TA",
    "Exter.Cond": "TA",
    "Foundation": "BrkTil",
    "Bsmt.Qual": "TA",
    "Bsmt.Cond": "TA",
    "Bsmt.Exposure": "No",
    "BsmtFin.Type.1": "Rec",
    "BsmtFin.SF.1": 381.0,
    "BsmtFin.Type.2": "Unf",
    "BsmtFin.SF.2": 0.0,
    "Bsmt.Unf.SF": 399.0,
    "Total.Bsmt.SF": 780.0,
    "Heating": "GasA",
    "Heating.QC": "Ex",
    "Central.Air": "Y",
    "Electrical": "SBrkr",
    "X1st.Flr.SF": 940,
    "X2nd.Flr.SF": 476,
    "Low.Qual.Fin.SF": 0,
    "Gr.Liv.Area": 1416,
    "Bsmt.Full.Bath": 0.0,
    "Bsmt.Half.Bath": 1.0,
    "Full.Bath": 1,
    "Half.Bath": 0,
    "Bedroom.AbvGr": 3,
    "Kitchen.AbvGr": 1,
    "Kitchen.Qual": "Gd",
    "TotRms.AbvGrd": 7,
    "Functional": "Typ",
    "Fireplaces": 0,
    "Fireplace.Qu": null,
    "Garage.Type": "Detchd",
    "Garage.Yr.Blt": 1956.0,
    "Garage.Finish": "Unf",
    "Garage.Cars": 2.0,
    "Garage.Area": 400.0,
    "Garage.Qual": "TA",
    "Garage.Cond": "TA",
    "Paved.Drive": "Y",
    "Wood.Deck.SF": 0,
    "Open.Porch.SF": 24,
    "Enclosed.Porch": 0,
    "X3Ssn.Porch": 0,
    "Screen.Porch": 0,
    "Pool.Area": 0,
    "Pool.QC": null,
    "Fence": null,
    "Misc.Feature": null,
    "Misc.Val": 0,
    "Mo.Sold": 6,
    "Yr.Sold": 2006,
    "Sale.Type": "WD ",
    "Sale.Condition": "Normal"
}

```
