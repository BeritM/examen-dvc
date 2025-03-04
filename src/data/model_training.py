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

#print(best_models.head(1))

best_model = best_models.iloc[0]["model"]

if best_model == "RandomForest":
    print("The selected model is 'Random Forest'")
    model = RandomForestRegressor()

elif best_model == "GradientBoosting":
    print("The selected model is 'Gradient Boosting'")
    model = GradientBoostingRegressor()

elif best_model == "SVR":
    print("The selected model is 'SVR'")
    model = SVR()


print("Training the model...")

model.fit(X_train, y_train)
joblib.dump(model, f"./models/trained_{best_model}.pkl")

print(f"Model saved: trained_{best_model}.pkl")
