import math
import numpy as np


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0

    def __init__(self, data_for_predict: np.ndarray, verification_data: np.ndarray, learning_rate=0.001):
        self.__data_for_predict = data_for_predict
        self.__verification_data = verification_data
        self.__training_set_size = verification_data.size
        self.__learning_rate = learning_rate
        self.__z_normalize_data()
        # self.__data_scaler()

    def __data_scaler(self):
        min_element = np.min(self.__data_for_predict)
        self.__data_for_predict += math.fabs(min_element)

        min_element = np.min(self.__verification_data)
        self.__verification_data += math.fabs(min_element)

    @staticmethod
    def __dispersion(data: np.ndarray):
        all_errors = (data - data.mean())
        all_errors = np.power(all_errors, 2)
        return all_errors.mean()

    def __z_normalize_data(self):
        # Нормализация __data_for_predict
        dispersion = math.sqrt(self.__dispersion(self.__data_for_predict))
        self.__data_for_predict = (self.__data_for_predict - self.__data_for_predict.mean())
        self.__data_for_predict /= dispersion

        # Нормализация __verification_data
        dispersion = math.sqrt(self.__dispersion(self.__verification_data))
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
        return middle_error

    def __calculation_k_1(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        all_errors *= self.__data_for_predict
        middle_error = all_errors.mean()
        return middle_error

    def training_model(self):
        # while self.__mse() > self.__accuracy:
        for i in range(20000):
            tmp_k_0 = self.k_0 - self.__learning_rate * self.__calculation_k_0()
            tmp_k_1 = self.k_1 - self.__learning_rate * self.__calculation_k_1()
            self.k_0 = tmp_k_0
            self.k_1 = tmp_k_1
            print('k0={}, k1={}, mse={}'.format(self.k_0, self.k_1, self.__mse()))
        return self.k_0, self.k_1
