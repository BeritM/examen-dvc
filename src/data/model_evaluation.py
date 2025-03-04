import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Load data...")

X_test = pd.read_csv("./data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("./data/processed_data/y_test.csv").values.ravel()

model = load("./models/trained_model.pkl")

print("Model prediction...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

eval_results = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2}

metrics_path = Path("./metrics/scores.json")
metrics_path.write_text(json.dumps(eval_results))

print(f"Evaluation saved to {metrics_path}")

