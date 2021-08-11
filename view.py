import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

traning_data = dict()


def my_print(k0, k1):
    # Парсинг всех точек датасета
    traning_set = genfromtxt('data.csv', delimiter=',')
    traning_set = np.delete(traning_set, 0, 0)
    traning_set = np.hsplit(traning_set, 2)
    data_for_predict = np.array(traning_set[0])
    verification_data = np.array(traning_set[1])

    # Отображения точек из датасета
    ax = plt.axes()
    ax.scatter(data_for_predict, verification_data, color='r')

    # Иментование осей
    ax.set_xlabel('millage')
    ax.set_ylabel('prise')

    # Используемые коэффициенты
    print('K0={}, K1={}'.format(k0, k1))

    # Отрисовка прямой
    plt.axline((0, k0), slope=k1, color='g')
    plt.show()



