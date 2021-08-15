import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from view import my_print
import linear_regression
import parse_csv

if __name__ == '__main__':
    # Парсинг датасета для обучения
    x_dataset, y_dataset = parse_csv.parse_csv()

    # Создание объекта класса линейной регрессии
    lr = linear_regression.LinearRegression(x_dataset, y_dataset)

    # Метод тренировки модели
    k0, k1 = lr.fit()

    # sklern z-standard
    x_standard_dataset = StandardScaler().fit_transform(x_dataset)

    print('my predict = {}'.format(lr.predict(x_dataset)))

    # Обучение модели и получение коэффициентов с помощью sklearn
    model = LinearRegression()
    model.fit(x_dataset, y_dataset)

    print('sklearn predict = {}'.format(model.predict(x_dataset)))

    # Создание (открытие) файла для сохранение полученных коэффицентов
    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(k0), float(k1)])

    # Графическое отображение
    my_print(float(k0), float(k1), float(model.intercept_), float(model.coef_), lr, model)
