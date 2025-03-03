import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.preprocessing import MinMaxScaler

def main(input_folder="./data/processed_data", output_folder="./data/processed_data"):
    """ Runs data processing script to split raw data from (../raw_data) into
        X_test, X_train, y_test and y_train (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('normalize features of the dataset')

    input_folder = "./data/processed_data"
    input_filepath_X_train = f"{input_folder}/X_train.csv"
    input_filepath_X_test = f"{input_folder}/X_test.csv"
    output_folder = input_folder

    data_processing(input_filepath_X_train, input_filepath_X_test, output_folder)

def data_processing(input_filepath_X_train, input_filepath_X_test, output_folder):
    X_train = import_dataset(input_filepath_X_train, sep=",")
    X_test = import_dataset(input_filepath_X_test, sep=",")
    X_train, X_test = convert_datetime(X_train, X_test)
    X_train_scaled, X_test_scaled = normalizer(X_train, X_test)    
    save_dataframe(X_train_scaled, X_test_scaled, output_folder)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)   

def convert_datetime(X_train, X_test):
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_test['date'] = pd.to_datetime(X_test['date'])
    #Convert datetime into unix timestamp
    X_train['date'] = X_train['date'].astype(int) / 10**9
    X_test['date'] = X_test['date'].astype(int) / 10**9
    return X_train, X_test

def normalizer(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def save_dataframe(X_train_scaled, X_test_scaled, output_folder):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folder, f'{filename}.csv')
        if os.path.exists(output_folder):
            pd.DataFrame(file).to_csv(output_filepath, index=False)
        else:
            print(f"Output folder {output_folder} doesn't exist.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()