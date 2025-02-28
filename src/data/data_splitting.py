import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
#from check_structure import check_existing_file, check_existing_folder
import os


@click.command()
@click.argument('input_folder', type=click.Path(exists=False), required=0)
@click.argument('output_folder', type=click.Path(exists=False), required=0)
def main(input_folder, output_folder):
    """ Runs data processing script to split raw data from (../raw_data) into
        X_test, X_train, y_test and y_train (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting data set into train and test datasets from raw data')

    input_folder = click.prompt('Enter the folder path for the input data', type=click.Path(exists=True))
    input_filepath = f"{input_folder}/raw.csv"
    output_folder = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    data_processing(input_filepath, output_folder)

def data_processing(input_filepath, output_folder):
    df = import_dataset(input_filepath, sep=",")
    X_train, X_test, y_train, y_test = split_data(df)
    save_dataframe(X_train, X_test, y_train, y_test, output_folder)

def import_dataset(filepath, **kwargs):
    return pd.read_csv(filepath, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def save_dataframe(X_train, X_test, y_train, y_test, output_folder):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folder, f'{filename}.csv')
        if os.path.exists(output_folder):
            file.to_csv(output_filepath, index=False)
        else:
            print("This output folder doesn't exist.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
