import numpy as np
import pandas as pd


def parse_csv():
    # Открыте файла с датасетом
    dataset = pd.read_csv('datasets/test_data.csv')

    x_dataset = dataset['x']
    y_dataset = dataset['y']

    x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)

    x_dataset = x_dataset.reshape(-1, 1)
    y_dataset = y_dataset.reshape(-1, 1)

    return x_dataset, y_dataset
