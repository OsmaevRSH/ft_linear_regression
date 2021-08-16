import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from parse_csv import parse_csv


def get_sklearn_predict_func(x, y, predict_value):
    sk = LinearRegression()
    sk.fit(x, y)
    pred_value = np.array(predict_value).reshape(-1, 1)
    return sk.intercept_, sk.coef_, sk.predict(pred_value)


def visualize(funk, predict_value, sklearn_visualize):
    predicted_y = k0 = k1 = 0.0
    # Парсинг всех точек датасета
    data_for_predict, verification_data = parse_csv()

    if sklearn_visualize:
        k0, k1, predicted_y = get_sklearn_predict_func(data_for_predict, verification_data, predict_value)

    fig = plt.figure()
    ax = plt.axes()

    plt.grid(b='true')

    # Отображения точек из датасета
    ax.scatter(data_for_predict, verification_data, color='g')
    ax.scatter(predict_value, funk(predict_value), color='r', s=200, label='ltheresi point')
    if sklearn_visualize:
        ax.scatter(predict_value, float(predicted_y), color='b', s=80, label='sklearn point')

    # Иментование осей
    ax.set_xlabel('independent variable')
    ax.set_ylabel('dependent variable')

    point = np.max(data_for_predict)
    x_max = float(point)
    y_max = float(funk(x_max))

    point = np.min(data_for_predict)
    x_min = float(point)
    y_min = float(funk(x_min))

    # Отрисовка моей линии
    plt.axline([x_min, y_min], [x_max, y_max], color='y', label='ltheresi liner regression')

    # Отрисовка линии полученной с помощью sklearn
    if sklearn_visualize:
        plt.axline((0, float(k0)), slope=float(k1), color='b', label='sklearn liner regression',
                   linestyle='dashed')

    # Отображение легенды
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center')

    # Отображения холста
    plt.show()
