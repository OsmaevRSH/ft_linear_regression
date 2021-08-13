import numpy as np
import pandas as pd


def parse_csv():
    # Открыте файла с датасетом
    df_train = pd.read_csv('datasets/test_data.csv')

    x_train = df_train['x']
    y_train = df_train['y']

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train
