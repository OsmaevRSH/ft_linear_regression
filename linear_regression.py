import numpy as np


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0
    __accuracy = 0.48

    def __init__(self, data_for_predict: np.ndarray, verification_data: np.ndarray, learning_rate=0.1):
        self.__data_for_predict = data_for_predict
        self.__verification_data = verification_data
        self.__training_set_size = verification_data.size
        self.__learning_rate = learning_rate

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
        while self.__mse() > self.__accuracy:
            test = self.__mse()
            for _ in range(self.__training_set_size):
                tmp_k_0 = self.k_0 - self.__calculation_k_0()
                tmp_k_1 = self.k_1 - self.__calculation_k_1()
                self.k_0 = tmp_k_0
                self.k_1 = tmp_k_1
                print('k0={}, k1={}, mse={}'.format(self.k_0, self.k_1, test))
        return self.k_0, self.k_1
