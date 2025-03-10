import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

def main(input_folder="./data/processed_data", output_folder="./models"):
    """Creates and process GridSearch with X_train, y_train and X_test, y_test (saved in ./raw/processed_data)
    and saves best parameters (in ./models).
    """
    
    input_filepath_X_train = f"{input_folder}/X_train_scaled.csv"
    input_filepath_y_train = f"{input_folder}/y_train.csv"

    X_train, y_train = load_data(input_filepath_X_train, input_filepath_y_train)
    perform_grid_search(X_train, y_train, output_folder)


    logger = logging.getLogger(__name__)
    logger.info('GridSearch')

    print("GridSearch finished!")

def load_data(input_filepath_X_train, input_filepath_y_train):
    """Loads X_train_scaled and y_train and converts y_train into 2-D-Array"""
    X_train = pd.read_csv(input_filepath_X_train)
    y_train = pd.read_csv(input_filepath_y_train).values.ravel() 
    return X_train, y_train

def perform_grid_search(X_train, y_train, output_folder):
    """ Performs GridSearch and saves best model.
    """
    print("Starting GridSearch")

    models = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 10],
            },
        },
        "SVR": {
            "model": SVR(),
            "params": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
        },
    }

    #best_models = {}
    results = []

    for model_name, model_info in models.items():
        print(f"\nStarting GridSearch for model: {model_name}")

        grid_search = GridSearchCV(
            estimator=model_info["model"],
            param_grid=model_info["params"],
            cv=5,  
            scoring="neg_mean_squared_error",
            n_jobs=-1,  
            verbose=2
        )

        # train model
        grid_search.fit(X_train, y_train)

        # save best model
        best_model = grid_search.best_estimator_
        model_path = f"{output_folder}/best_model_{model_name}.pkl"
        joblib.dump(best_model, model_path)

        # Save best hyperparameters
        print(f"Best hyperparameters for model {model_name}: {grid_search.best_params_}")
        print(f"Best neg. MSE for model {model_name}: {grid_search.best_score_}")
        #best_models[model_name] = grid_search.best_params_

        results.append({
            "model": model_name,
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_
        })

    results_df = pd.DataFrame(results).sort_values(by="best_score", ascending=False)
    results_df.to_csv(f"{output_folder}/best_models.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model_path = f"{output_folder}/best_model_{best_model_name}.pkl"
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, f"{output_folder}/final_best_model.pkl")

    print("\nBest model and parameters:")
    print(results_df.head(1))

    return results_df

    #return best_models


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()


