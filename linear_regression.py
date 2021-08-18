import math
import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, learning_rate=0.1):
        """
        Конструктор
        :param x_dataset: Массив независимых переменных
        :param y_dataset: Массив зависимых переменных
        :param learning_rate: Скорость обучения
        """
        self.__x_dataset = x_dataset
        self.__y_dataset = y_dataset
        self.__learning_rate = learning_rate
        self.x_mean = 0
        self.x_dispersion = 0
        self.k_0 = np.zeros(1).reshape(-1, 1)
        self.k_1 = np.zeros(1).reshape(-1, 1)
        self.__mse_error = np.zeros(1).reshape(-1, 1)
        self.__z_standardization()

    def __dispersion(self):
        """
        Метод для расчета дисперсии
        :return: Дисперсия __x_dataset
        """
        all_errors = (self.__x_dataset - self.__x_dataset.mean())
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __z_standardization(self):
        """
        Метод z-стандартизации для __x_dataset
        :return: Стандартизированный датасет
        """
        self.x_dispersion = math.sqrt(self.__dispersion())
        self.x_mean = self.__x_dataset.mean()
        self.__x_dataset = (self.__x_dataset - self.x_mean)
        self.__x_dataset /= self.x_dispersion

    def __predict_price(self, x):
        """
        Метод для предсказания цены
        :param x: Независимая переменная, по которой происходит предсказание
        :return: Предсказанная цена
        """
        predicted_price = self.k_0[-1] + self.k_1[-1] * x
        return predicted_price

    def __mse(self):
        """
        Метод для расчета среднеквадратичной ошибки (Функция потерь)
        :return: Среднеквадратичная ошибка
        """
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __error(self, x, y):
        """
        Метод для рассчета ошибки предсказания
        :param x: Независимая переменная
        :param y: Зависимая переменная
        :return: Ошибка
        """
        error = self.__predict_price(x) - y
        return error

    def __calculation_k_0(self):
        """
        Градиентный спуск для расчета свободного члена
        """
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        middle_error = all_errors.mean()
        return middle_error

    def __calculation_k_1(self):
        """
        Градиентный спуск для расчета члена перед независимой переменной
        """
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        all_errors *= self.__x_dataset
        middle_error = all_errors.mean()
        return middle_error

    def fit(self, logging_status):
        """
        Метод обучения модели
        """
        delta_mse = 1
        old_mse = self.__mse()
        population = 1
        while math.fabs(delta_mse) > 0.000001:
            tmp_k_0 = self.__learning_rate * self.__calculation_k_0()
            tmp_k_1 = self.__learning_rate * self.__calculation_k_1()
            self.k_0 = np.append(self.k_0, self.k_0[-1] - tmp_k_0)
            self.k_1 = np.append(self.k_1, self.k_1[-1] - tmp_k_1)
            delta_mse = old_mse - self.__mse()
            old_mse = self.__mse()
            self.__mse_error = np.append(self.__mse_error, self.__mse())
            if logging_status:
                print('population={}, k0={}, k1={}, mse={}, learningRate={}'
                      .format(population, self.k_0[-1], self.k_1[-1], self.__mse(), self.__learning_rate))
                population += 1

        df = pd.DataFrame({'k_0': self.k_0,
                           'k_1': self.k_1})
        df.to_csv('coefficients.csv', index=False)

        df = pd.DataFrame({'mse': self.__mse_error})
        df.to_csv('mse.csv', index=False)
