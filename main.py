import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from view import my_print
import model_training
import parse_csv

if __name__ == '__main__':
    # Парсинг датасета для обучения
    data_for_predict, verification_data = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = model_training.LinearRegression(data_for_predict, verification_data)

    # Метод тренировки модели
    k0, k1 = lr.training_model()

    # Формировние данных из датасета для sklearn
    x = np.array(data_for_predict).reshape((-1, 1))
    y = np.array(verification_data)

    # Обучение модели и получение коэффициентов с помощью sklearn
    model = LinearRegression().fit(x, y)

    # Создание (открытие) файлы для сохранение полученных коэффицентов
    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(k0), float(k1)])

    # Графическое отображение
    my_print(float(k0), float(k1), float(model.intercept_), float(model.coef_))
