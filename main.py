import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  # pip install xgboost

def load_data():
    data = datasets.fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    pipelines = {}
    for name, model in models.items():
        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            pipelines[name] = Pipeline([('model', model)])
        else:
            pipelines[name] = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
    return pipelines

def evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    results = []

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append((name, mse, mae, r2))

        print(f"\n{name}")
        print(f"  MSE = {mse:.3f}")
        print(f"  MAE = {mae:.3f}")
        print(f"  R² Score = {r2:.3f}")

        # Visualization: Prediction vs. Reality
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} - Prediction")
        plt.grid(True)
        plt.show()

        # Visualization: Residual Graph
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title(f"{name} - Residuals")
        plt.grid(True)
        plt.show()

    return results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    pipelines = get_models()
    results = evaluate_models(pipelines, X_train, X_test, y_train, y_test)
