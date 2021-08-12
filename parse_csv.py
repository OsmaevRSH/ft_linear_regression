import numpy as np
from numpy import genfromtxt


def parse_csv():
    # Открыте файла с датасетом
    training_set = genfromtxt('test_data.csv', delimiter=',')

    # Удаление первой строки с названием столбцов
    training_set = np.delete(training_set, 0, 0)

    # Деление одного двумерного массива на два одномерных по вериткали
    training_set = np.hsplit(training_set, 2)

    # Сохранения данных на основе которых будет происходить предсказание
    data_for_predict = np.array(training_set[0])

    # Сохранение данных, с которыми будут сравниваться предсказанные значения
    verification_data = np.array(training_set[1])

    return data_for_predict, verification_data
