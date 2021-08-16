import sys
import pandas as pd
import linear_regression
import parse_csv


def train():
    # Парсинг датасета для обучения
    x_dataset, y_dataset = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = linear_regression.LinearRegression(x_dataset, y_dataset)

    # Метод тренировки модели
    lr.fit(logging_status)

    df = pd.DataFrame({'k_0': [lr.k_0[-1]],
                       'k_1': [lr.k_1[-1]],
                       'dispersion': [lr.x_dispersion],
                       'mean': [lr.x_mean]})

    df.to_csv('predict_coefficients.csv', index=False)


if __name__ == '__main__':
    logging_status = False

    if len(sys.argv) - 1 > 0:
        if sys.argv[1] == '--log':
            logging_status = True
    train()
