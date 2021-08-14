import math
import numpy as np


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, learning_rate=0.1):
        self.__x_dataset = x_dataset
        self.__y_dataset = y_dataset
        self.__training_set_size = y_dataset.size
        self.__learning_rate = learning_rate
        self.__z_normalize_data()
        self.__data_scaler()

    def __data_scaler(self):
        min_element = np.min(self.__x_dataset)
        max_element = np.max(self.__x_dataset)
        self.__x_dataset = (self.__x_dataset.astype(float) - min_element) / (max_element - min_element)

        min_element = np.min(self.__y_dataset)
        max_element = np.max(self.__y_dataset)
        self.__y_dataset = (self.__y_dataset.astype(float) - min_element) / (max_element - min_element)

    @staticmethod
    def __dispersion(data: np.ndarray):
        all_errors = (data - data.mean())
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __z_normalize_data(self):
        # Нормализация __x_dataset
        dispersion = math.sqrt(self.__dispersion(self.__x_dataset))
        self.__x_dataset = (self.__x_dataset - self.__x_dataset.mean())
        self.__x_dataset /= dispersion

        # Нормализация __y_dataset
        dispersion = math.sqrt(self.__dispersion(self.__y_dataset))
        self.__y_dataset = (self.__y_dataset - self.__y_dataset.mean())
        self.__y_dataset /= dispersion

    def predict_price(self, x):
        predicted_price = self.k_0 + self.k_1 * x.astype(float)
        return predicted_price

    def __mse(self):
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __error(self, x, y):
        error = self.predict_price(x) - y.astype(float)
        return error

    def __calculation_k_0(self):
        all_errors = self.__error(self.__x_dataset.astype(float), self.__y_dataset.astype(float))
        middle_error = all_errors.mean()
        return middle_error

    def __calculation_k_1(self):
        all_errors = self.__error(self.__x_dataset.astype(float), self.__y_dataset.astype(float))
        all_errors *= self.__x_dataset.astype(float)
        middle_error = all_errors.mean()
        return middle_error

    def training_model(self):
        # while self.__mse() > self.__accuracy:
        for i in range(2000):
            tmp_k_0 = self.__learning_rate * self.__calculation_k_0()
            tmp_k_1 = self.__learning_rate * self.__calculation_k_1()
            self.k_0 -= tmp_k_0
            self.k_1 -= tmp_k_1
            print('k0={}, k1={}, mse={}'.format(self.k_0, self.k_1, self.__mse()))
        return self.k_0, self.k_1
