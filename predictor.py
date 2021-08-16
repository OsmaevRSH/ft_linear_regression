import sys

import numpy as np
import pandas as pd

from parse_csv import parse_csv
from visualize import visualize, visualize_train


class Predictor:
    def __init__(self):
        """
        Конструктор
        """
        self.__predicted_value = 0.0
        self.__k_0 = 0.0
        self.__k_1 = 0.0
        self.__dispersion = 0.0
        self.__mean = 0.0
        self.__parse_coefficients()

    def __parse_coefficients(self):
        """
        Метод для парсинга коэффициентов полученных после тренировки модели
        """
        try:
            coefficients = pd.read_csv('predict_coefficients.csv')
            self.__k_0 = float(coefficients['k_0'])
            self.__k_1 = float(coefficients['k_1'])
            self.__dispersion = float(coefficients['dispersion'])
            self.__mean = float(coefficients['mean'])
        except pd.errors.EmptyDataError:
            raise Exception('No columns to parse from file!')
        except TypeError:
            raise Exception('There are no coefficients in the file to calculate, or the data type is wrong!')
        except FileNotFoundError:
            raise Exception('File with coefficients not found, run training.py!')
        except Exception:
            raise

    def predict(self, data):
        """
        Метод для средсказания зависимой переменной
        :param data: Независимая переменная, на основе которой происходит предсказание
        :return: Предсказанное значение
        """
        try:
            self.__predicted_value = self.__k_0 + ((data - self.__mean) / self.__dispersion) * self.__k_1
        except Exception:
            raise
        else:
            return self.__predicted_value

    def visualize_data(self):
        """
        Метод визуализации
        """
        visualize(self.predict, value_for_predict, sklearn_visualize)

    def visualize_train(self):
        """
        Метод визуализации
        """
        visualize_train(self.__mean, self.__dispersion)

    def r2(self):
        x, y = parse_csv()
        r_2 = np.sum(np.power(self.predict(x) - y.mean(), 2)) / np.sum(np.power(y - y.mean(), 2))
        return r_2


if __name__ == '__main__':
    visualize_data = False
    sklearn_visualize = False
    train_visualize = False
    r2 = False

    if len(sys.argv) - 1 > 0:
        if sys.argv.__contains__('--visualize'):
            visualize_data = True
        if sys.argv.__contains__('--visualize_sk'):
            sklearn_visualize = True
        if sys.argv.__contains__('--visualize_train'):
            train_visualize = True
        if sys.argv.__contains__('--r2'):
            r2 = True
    try:
        pr = Predictor()
        if train_visualize:
            pr.visualize_train()
            exit(0)
        if r2:
            print('R2 = {}'.format(pr.r2()))
            exit(0)
        value_for_predict = float(input('Enter value for predict: '))
        predicted_value = pr.predict(value_for_predict)
        print('Predicted data = {}'.format(predicted_value))
        if predicted_value != 0 and visualize_data:
            pr.visualize_data()
    except Exception as e:
        print(str(e))
