import math
import numpy as np


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0
    __mse_error = np.zeros(1).reshape(-1, 1)

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, learning_rate=0.1):
        self.__x_dataset = x_dataset
        self.__y_dataset = y_dataset
        self.__learning_rate = learning_rate
        self.__x_dataset = self.__z_standartization(x_dataset)
        # self.__normalization()

    @staticmethod
    def __dispersion(data: np.ndarray):
        all_errors = (data - data.mean())
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __z_standartization(self, dataset):
        dispersion = math.sqrt(self.__dispersion(dataset))
        dataset = (dataset - dataset.mean())
        dataset /= dispersion
        return dataset

    def __normalization(self):
        min_x = np.min(self.__x_dataset)
        max_x = np.max(self.__x_dataset)
        self.__x_dataset = (self.__x_dataset - min_x) / (max_x - min_x)

    def __predict_price(self, x):
        predicted_price = self.k_0 + self.k_1 * x
        return predicted_price

    def __mse(self):
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __error(self, x, y):
        error = self.__predict_price(x) - y
        return error

    def __calculation_k_0(self):
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        middle_error = all_errors.mean()
        return middle_error

    def __calculation_k_1(self):
        all_errors = self.__error(self.__x_dataset, self.__y_dataset)
        all_errors *= self.__x_dataset
        middle_error = all_errors.mean()
        return middle_error

    def training_model(self):
        delta_mse = 1
        old_mse = self.__mse()
        population = 1
        while math.fabs(delta_mse) > 0.000001:
            tmp_k_0 = self.__learning_rate * self.__calculation_k_0()
            tmp_k_1 = self.__learning_rate * self.__calculation_k_1()
            self.k_0 -= tmp_k_0
            self.k_1 -= tmp_k_1
            delta_mse = old_mse - self.__mse()
            old_mse = self.__mse()
            self.__mse_error = np.append(self.__mse_error, self.__mse())

            print('population={}, k0={}, k1={}, mse={}, learningRate={}'
                  .format(population, self.k_0, self.k_1, self.__mse(), self.__learning_rate))
            population += 1
        return self.k_0, self.k_1

    def predict(self, x):
        return self.k_0 + (self.__z_standartization(x) * self.k_1)
