import numpy as np
from numpy import genfromtxt


def parse_csv():
    # Открыте файла с датасетом
    traning_set = genfromtxt('data.csv', delimiter=',')

    # Удаление первой строки с названием столбцов
    traning_set = np.delete(traning_set, 0, 0)

    # Деление одного двумерного массива на два одномерных по вериткали
    traning_set = np.hsplit(traning_set, 2)

    # Сохранения данных на основе которых будет происходить предсказание
    data_for_predict = np.array(traning_set[0])

    # Сохранение данных, с которыми будут сравниваться предсказанные значения
    verification_data = np.array(traning_set[1])

    return data_for_predict, verification_data
