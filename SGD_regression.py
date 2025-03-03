import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset (Replace with actual dataset path or method)
df = pd.read_csv("housing.csv")

# Display dataset columns to verify target column name
print("Columns in dataset:", df.columns)

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# Define features and target (Replace 'median_house_value' with real target column name)
X = df.drop(columns=["median_house_value", "ocean_proximity"])  # Dropping categorical feature for now
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and SGD Regressor
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

# Define hyperparameter grid
param_grid = {
    "sgd__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "sgd__penalty": ["l2", "l1", "elasticnet"]
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
