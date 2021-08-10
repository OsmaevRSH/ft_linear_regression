# Пример 1.4.1

import matplotlib.pyplot as plt

from parse_csv import traning_data


def my_print(k0, k1):
    fig = plt.figure()  # Создание объекта Figure
    print(fig.axes)  # Список текущих областей рисования пуст
    print(type(fig))  # тип объекта Figure
    for key in traning_data:
        plt.scatter(key, traning_data[key])  # scatter - метод для нанесения маркера в точке (1.0, 1.0)
    # y = k1x + k0
    plt.axline((0, 0), slope=k1, color='r')

    # После нанесения графического элемента в виде маркера
    # список текущих областей состоит из одной области
    print(fig.axes)

    plt.show()
