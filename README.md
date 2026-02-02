# IA-ML - Governance and Public Administration

This repository contains scripts for data preprocessing, exploratory data analysis, and Machine Learning models (Linear Regression, Random Forest, and Decision Tree) applied to a public governance and administration dataset.

The main objective is to analyze operational indicators and predict **citizen satisfaction**.

---

## Project Structure (example)

- `dataset_governanca_admin_publica.csv` - original dataset  
- `dataset_governanca_admin_publica_preprocessado.csv` - preprocessed dataset  
- `preprocessamento.py` - data cleaning, missing values handling, normalization, and date conversion  
- `analise_exploratoria.py` - exploratory analysis (histograms, boxplots, correlation heatmap)  
- `regressao_satisfacao.py` - Linear Regression model to predict `satisfacao_cidadao`  
- `regressao_randomforest.py` - Random Forest Regressor  
- `arvore_decisao_satisfacao.py` - Decision Tree Regressor with image export  

> File names may vary slightly depending on project organization, but this README assumes they are located in the main project folder.

---

## Requirements

- Python 3.9 or higher (recommended)
- Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Install dependencies

Run the following command in the project directory:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run the Project
1) Data Preprocessing

Generates the preprocessed dataset:

python preprocessamento.py

Output:

dataset_governanca_admin_publica_preprocessado.csv

2) Exploratory Data Analysis

Displays distributions, boxplots, and correlation heatmaps:

python analise_exploratoria.py

In environments without graphical interface, plots can be saved using plt.savefig().

3) Linear Regression (Citizen Satisfaction)
python regressao_satisfacao.py

Output:

Evaluation metrics in the terminal (MSE, MAE, R²)

resultados_regressao_satisfacao.csv

4) Random Forest Regression
python regressao_randomforest.py

Output:

Evaluation metrics in the terminal (MSE, MAE, R²)

resultados_regressao_randomforest.csv

5) Decision Tree Regression
python arvore_decisao_satisfacao.py

Output:

Evaluation metrics in the terminal (MSE, MAE, R²)

resultados_arvore_decisao.csv — real vs predicted values