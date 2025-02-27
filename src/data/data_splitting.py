import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing script to split raw data from (../raw_data) into
        X_test, X_train, y_test and y_train (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting data set into train and test datasets from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_csv = f"{input_filepath}/raw.csv"
    output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    splitter(input_filepath_csv, output_filepath)

def splitter(input_filepath_csv, output_filepath):
