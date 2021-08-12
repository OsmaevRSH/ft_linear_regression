import math
import numpy as np


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0
    # __accuracy = 0.0000000001

    def __init__(self, data_for_predict: np.ndarray, verification_data: np.ndarray, learning_rate=0.1):
        self.__data_for_predict = data_for_predict
        self.__verification_data = verification_data
        self.__training_set_size = verification_data.size
        self.__learning_rate = learning_rate
        self.__z_normalize_data()

    def __z_mse(self, data: np.ndarray):
        all_errors = (data - data.mean())
        all_errors = np.power(all_errors, 2)
        return np.sum(all_errors) / (self.__training_set_size - 1)

    def __z_normalize_data(self):
        dispersion = math.sqrt(self.__z_mse(self.__data_for_predict))
        self.__data_for_predict = (self.__data_for_predict - self.__data_for_predict.mean())
        self.__data_for_predict /= dispersion

        dispersion = math.sqrt(self.__z_mse(self.__verification_data))
        self.__verification_data = (self.__verification_data - self.__verification_data.mean())
        self.__verification_data /= dispersion

    def predict_price(self, millage):
        predicted_price = self.k_0 + self.k_1 * millage
        return predicted_price

    def __mse(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __error(self, millage, price):
        error = self.predict_price(millage) - price
        return error

    def __calculation_k_0(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        middle_error = all_errors.mean()
        return self.__learning_rate * middle_error

    def __calculation_k_1(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        all_errors *= self.__data_for_predict
        middle_error = all_errors.mean()
        return self.__learning_rate * middle_error

    def training_model(self):
        # while self.__mse() > self.__accuracy:
        for i in range(200000):
            tmp_k_0 = self.k_0 - self.__calculation_k_0()
            tmp_k_1 = self.k_1 - self.__calculation_k_1()
            self.k_0 = tmp_k_0
            self.k_1 = tmp_k_1
            print('k0={}, k1={}, mse={}'.format(self.k_0, self.k_1, self.__mse()))
        return self.k_0, self.k_1
