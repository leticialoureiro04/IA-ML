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

To run this project, the following software and libraries are required:

- **Python** 3.9 or higher (recommended)
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **ydata-profiling** (for the EDA HTML report)

### Optional
- A web browser (e.g., Chrome, Firefox) to view the generated HTML report (`report.html`)

### Installation

Install the required dependencies by running the following command in the project directory:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

## How to Run the Project

### 1) Data Preprocessing

Generates the preprocessed dataset used in all subsequent analyses.

```bash
python preprocessamento.py

**Output:**
- `dataset_governanca_admin_publica_preprocessado.csv`

### 2) Exploratory Data Analysis

Displays distributions, boxplots, and correlation heatmaps.

```bash
python analise_exploratoria.py

**Notes:**
- In environments without a graphical interface, plots can be saved using `plt.savefig()`.

### 3) Linear Regression (Citizen Satisfaction)

Trains a Linear Regression model to predict citizen satisfaction.

```bash
python regressao_satisfacao.py

**Output:**
- Evaluation metrics printed in the terminal (MSE, MAE, R²)
- `resultados_regressao_satisfacao.csv`

### 4) Random Forest Regression

Trains a Random Forest Regressor for predicting citizen satisfaction.

```bash
python regressao_randomforest.py

**Output:**
- Evaluation metrics printed in the terminal (MSE, MAE, R²)
- `resultados_regressao_randomforest.csv`

### 5) Decision Tree Regression

Trains a Decision Tree Regressor and exports the tree visualization.

```bash
python arvore_decisao_satisfacao.py

**Output:**
- Evaluation metrics printed in the terminal (MSE, MAE, R²)
- `resultados_arvore_decisao.csv` — real vs. predicted values

## Exploratory Data Analysis Report (EDA)

An interactive exploratory data analysis report generated with **ydata-profiling**
is available online:

👉 https://leticialoureiro04.github.io/IA-ML/report.html