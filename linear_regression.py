import math
import numpy as np


class LinearRegression:
    __k_0 = 0.0
    __k_1 = 0.0
    __mse_error = np.zeros(1).reshape(-1, 1)

    def __init__(self, x_dataset: np.ndarray, y_dataset: np.ndarray, learning_rate=0.1):
        self.__x_dataset = x_dataset
        self.__y_dataset = y_dataset
        self.__learning_rate = learning_rate
        self.__x_mean = 0
        self.__x_dispersion = 0
        self.__z_standardization()

    @staticmethod
    def __dispersion(data: np.ndarray):
        all_errors = (data - data.mean())
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __z_standardization(self):
        self.__x_dispersion = math.sqrt(self.__dispersion(self.__x_dataset))
        self.__x_mean = self.__x_dataset.mean()
        self.__x_dataset = (self.__x_dataset - self.__x_mean)
        self.__x_dataset /= self.__x_dispersion

    def __predict_price(self, x):
        predicted_price = self.__k_0 + self.__k_1 * x
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
            self.__k_0 -= tmp_k_0
            self.__k_1 -= tmp_k_1
            delta_mse = old_mse - self.__mse()
            old_mse = self.__mse()
            self.__mse_error = np.append(self.__mse_error, self.__mse())

            print('population={}, k0={}, k1={}, mse={}, learningRate={}'
                  .format(population, self.__k_0, self.__k_1, self.__mse(), self.__learning_rate))
            population += 1
        return self.__k_0, self.__k_1

    def predict(self, x):
        return self.__k_0 + (((x - self.__x_mean) / self.__x_dispersion) * self.__k_1)
