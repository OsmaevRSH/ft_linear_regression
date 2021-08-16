import csv
import sys
import linear_regression
import parse_csv


def train():
    # Парсинг датасета для обучения
    x_dataset, y_dataset = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = linear_regression.LinearRegression(x_dataset, y_dataset)

    # Метод тренировки модели
    lr.fit(logging_status)

    # Создание (открытие) файла для сохранение полученных коэффицентов
    with open('coefficients.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(lr.k_0), float(lr.k_1)])


if __name__ == '__main__':
    logging_status = False

    if len(sys.argv) - 1 > 0:
        if sys.argv[1] == '--log':
            logging_status = True
    train()
