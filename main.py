import csv
from sklearn.linear_model import LinearRegression
from view import my_print
import linear_regression
import parse_csv

if __name__ == '__main__':
    # Парсинг датасета для обучения
    data_for_predict, verification_data = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = linear_regression.LinearRegression(data_for_predict, verification_data)

    # Метод тренировки модели
    k0, k1 = lr.training_model()

    # Обучение модели и получение коэффициентов с помощью sklearn
    model = LinearRegression().fit(data_for_predict, verification_data)

    # Создание (открытие) файлы для сохранение полученных коэффицентов
    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(k0), float(k1)])

    # Графическое отображение
    my_print(float(k0), float(k1), float(model.intercept_), float(model.coef_))
