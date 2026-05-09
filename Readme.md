# Analysis and Prediction of Car Fuel Consumption

**Student:** Dimitar Dimitrov (201223056)

## Project Overview
This project applies Machine Learning techniques to analyze vehicle specifications and predict $CO_2$ emissions. By using features like engine size, cylinder count, and combined fuel consumption, the models provide highly accurate environmental impact assessments.

## Features
- **Exploratory Data Analysis (EDA):** Visualizing correlations using Seaborn heatmaps.
- **Data Preprocessing:** Handling missing values and feature scaling using `StandardScaler`.
- **Multiple Models:** Comparison between Linear Regression, Ridge, and Gradient Boosting.
- **High Accuracy:** The Gradient Boosting model achieves an $R^2$ score of ~0.96.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

Run the analysis 
python fuel_analysis.py

## Technological Stack
- **Language:** Python 3.10+
- **Key Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn.