import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
import os

def main(input_folder, output_folder):
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
    df_X_train = import_dataset(input_filepath_X_train, sep=",")
    df_X_test = import_dataset(input_filepath_X_test, sep=",")
    

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)    
