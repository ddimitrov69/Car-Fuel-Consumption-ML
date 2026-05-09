"""
Project: Analysis and Prediction of Car Fuel Consumption and Emissions
Student: Dimitar Dimitrov (201223056)
Description: Machine Learning pipeline for predicting CO2 emissions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 1. DATA LOADING & INITIAL EXPLORATION
# ==========================================
def load_data(file_path):
    # Loading the dataset (FuelConsumption.csv)
    df = pd.read_csv(file_path)
    print("--- Dataset Loaded ---")
    print(df.head())
    return df

# ==========================================
# 2. STATISTICAL ANALYSIS & VISUALIZATION (2B)
# ==========================================
def perform_eda(df):
    print("\n--- Performing Statistical Analysis ---")
    
    # Selecting numerical columns for correlation
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title('Feature Correlation Matrix (Heatmap)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png') # For GitHub/Presentation
    plt.show()

    # Scatter plot: Engine Size vs CO2 Emissions
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df, color='teal')
    plt.title('Impact of Engine Size on CO2 Emissions')
    plt.savefig('engine_vs_co2.png')
    plt.show()

# ==========================================
# 3. PREPROCESSING (2C)
# ==========================================
def preprocess_data(df):
    print("\n--- Preprocessing Dataset ---")
    
    # Selecting Features (ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB)
    # Target (CO2EMISSIONS)
    X = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
    y = df['CO2EMISSIONS']
    
    # Handling potential noise/missing data
    X = X.fillna(X.median())
    
    # Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling: Normalize data for better algorithm convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# 4. ALGORITHMS & MODEL SYNTHESIS (4 & 5)
# ==========================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # Model 1: Multiple Linear Regression (Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # Model 2: Ridge Regression (L2 Regularization)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    # Model 3: Gradient Boosting Regressor (Advanced Ensemble)
    gbm = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)

    # Metrics Compilation
    models = {"Linear Regression": lr_pred, "Ridge": ridge_pred, "Gradient Boosting": gbm_pred}
    
    print("\n--- Experimental Results ---")
    for name, preds in models.items():
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results[name] = r2
        print(f"{name} -> R2 Score: {r2:.4f}, MAE: {mae:.2f}")
        
    return results

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    
        data = load_data('FuelConsumption.csv')
        perform_eda(data)
        X_tr, X_te, y_tr, y_te = preprocess_data(data)
        metrics = train_and_evaluate(X_tr, X_te, y_tr, y_te)
        
        # Comparative Visualization of results
        plt.figure(figsize=(10, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['gray', 'blue', 'green'])
        plt.ylabel('R2 Score (Higher is Better)')
        plt.title('Model Performance Comparison')
        plt.ylim(0.8, 1.0)
        plt.show()
        
    except FileNotFoundError:
        print("Error: Please provide 'FuelConsumption.csv' in the same folder.")