import csv
import linear_regression
import parse_csv


def train():

    # todo Добавиль включение и отключение лога

    # Парсинг датасета для обучения
    x_dataset, y_dataset = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = linear_regression.LinearRegression(x_dataset, y_dataset)

    # Метод тренировки модели
    k0, k1 = lr.fit()

    # Создание (открытие) файла для сохранение полученных коэффицентов
    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(k0), float(k1)])


if __name__ == '__main__':
    train()
