import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
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


def visualize_train(mean, dispersion):
    data_for_predict, verification_data = parse_csv()
    coefficients = pd.read_csv('coefficients.csv')
    k_0 = np.array(coefficients['k_0'])
    k_1 = np.array(coefficients['k_1'])

    fig, ax = plt.subplots()
    plt.grid(b='true')
    ax.scatter(data_for_predict, verification_data, color='g')
    ax.set_xlabel('independent variable')
    ax.set_ylabel('dependent variable')

    plt.subplots_adjust(bottom=0.25)

    slider_color = 'White'

    axis_position = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=slider_color)

    slider_position = Slider(axis_position, label='Epoch', valmin=0, valmax=k_0.size, valinit=0, valstep=1)

    def get_new_data(data, index):
        if index >= k_0.size:
            index = k_0.size - 1
        if dispersion == 0:
            return 0
        return k_0[int(index)] + ((data - mean) / dispersion) * k_1[int(index)]

    def update(val):
        ax.clear()
        ax.grid(b='true')
        ax.scatter(data_for_predict, verification_data, color='g')
        ax.set_xlabel('independent variable')
        ax.set_ylabel('dependent variable')

        pos = slider_position.val
        point = np.max(data_for_predict)
        x_max = float(point)
        y_max = float(get_new_data(x_max, pos))

        point = np.min(data_for_predict)
        x_min = float(point)
        y_min = float(get_new_data(x_min, pos))

        ax.axline([x_min, y_min], [x_max, y_max], color='y', label='ltheresi liner regression')
        fig.canvas.draw_idle()

    update(0)

    slider_position.on_changed(update)

    plt.show()


def visualize_mse():
    mse = pd.read_csv('mse.csv')
    errors = np.array(mse['mse']).flatten()
    indexes = np.arange(start=0, stop=errors.size, step=1)
    plot_array = np.column_stack([indexes[1:], errors[1:]])
    fig = plt.figure()
    ax = plt.axes()

    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')

    ax.grid(b='true')

    plt.plot(plot_array[:, 0], plot_array[:, 1], '-b')

    plt.show()
