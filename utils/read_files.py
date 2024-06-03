import pandas as pd
import yaml

RATINGS_PATH = './data/preprocessed_data/ratings/ratings_preprocessed_ml.csv'
JOKES_PATH = './data/preprocessed_data/jokes/jokes_preprocessed.csv'


def read_yaml_file():
    with open('neumf-config.yaml', "r") as file:
        data = yaml.safe_load(file)
    return data

def read_jokes_csv_file():
    return pd.read_csv(JOKES_PATH, sep=',', header=0)


def read_ratings_csv_file():
    return pd.read_csv(RATINGS_PATH, sep=',', header=0)