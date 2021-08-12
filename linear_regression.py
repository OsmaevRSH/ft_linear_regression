import multiprocessing

import numpy as np
from multiprocessing import Process


class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0
    __accuracy = 0.48
    __procs = []

    def __init__(self, data_for_predict: np.ndarray, verification_data: np.ndarray, learning_rate=0.00000000015):
        self.__data_for_predict = data_for_predict
        self.__verification_data = verification_data
        self.__training_set_size = verification_data.size
        self.__learning_rate = learning_rate

    def __parallel_run(self):
        tmp_k_0 = multiprocessing.Value("f", 0.0, lock=False)
        tmp_k_1 = multiprocessing.Value("f", 0.0, lock=False)
        p1 = Process(target=self.__async_calculation_k_0, args=[tmp_k_0])
        p2 = Process(target=self.__async_calculation_k_1, args=[tmp_k_1])
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        return tmp_k_0.value, tmp_k_1.value

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

    def __async_calculation_k_0(self, tmp_k_0):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        middle_error = all_errors.mean()
        tmp_k_0.value = self.__learning_rate * middle_error

    def __async_calculation_k_1(self, tmp_k_1):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data) * self.__data_for_predict
        middle_error = all_errors.mean()
        tmp_k_1.value = self.__learning_rate * middle_error

    def __calculation_k_0(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data)
        middle_error = all_errors.mean()
        return self.__learning_rate * middle_error

    def __calculation_k_1(self):
        all_errors = self.__error(self.__data_for_predict, self.__verification_data) * self.__data_for_predict
        middle_error = all_errors.mean()
        return self.__learning_rate * middle_error

    def training_model(self):
        while self.__mse() > self.__accuracy:
            for _ in range(self.__training_set_size):
                tmp_k_0 = self.k_0 - self.__calculation_k_0()
                tmp_k_1 = self.k_1 - self.__calculation_k_1()
                self.k_0 = tmp_k_0
                self.k_1 = tmp_k_1

                # multiprocessing
                # tmp_k_0, tmp_k_1 = self.__parallel_run()
                # self.k_0 -= tmp_k_0
                # self.k_1 -= tmp_k_1

                print(self.k_0, self.k_1)
        return self.k_0, self.k_1
