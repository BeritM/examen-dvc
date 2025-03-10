import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

print("Load data...")

X_train = pd.read_csv("./data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("./data/processed_data/y_train.csv").values.ravel()
best_models = pd.read_csv("./models/best_models.csv")
model = joblib.load("./models/final_best_model.pkl")

best_model = best_models.iloc[0]["model"]
print(f"The selected model is {best_model}")

print("Training the model...")
model.fit(X_train, y_train)

joblib.dump(model, f"./models/trained_model.pkl")

print(f"Model saved: trained_model.pkl")
