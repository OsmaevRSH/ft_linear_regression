class LinearRegression:
    k_0 = 0.0
    k_1 = 0.0

    def __init__(self, traning_data: dict, learning_rate=0.01):
        """
        Конструктор
        :param traning_data: словарь для обучения модели [millage, price]
        :param learning_rate: скорость обучения
        """
        self.__traning_data = traning_data
        self.__learning_rate = learning_rate

    def predict_price(self, millage):
        """
        Метод для предсказания цены
        :param millage: Текущий пробег
        :return: Предсказанная цена
        """
        predicted_price = self.k_0 + self.k_1 * millage
        return predicted_price

    def __error(self, millage, price):
        """
        Метод для расчета ошибки предсказания
        :param millage: Текущий пробег
        :param price: Ожидаемая цена
        :return: Ошибка
        """
        error = self.predict_price(millage) - price
        return error

    def __calculation_k_0(self):
        """
        Метод расчета нулевого (свободного) коэффициента
        :return: Расчитанный нулевой коэффициент
        """
        result = 0
        for key in self.__traning_data:
            result += self.__error(key, self.__traning_data[key])
        middle_error = result / self.__traning_data.__len__()
        return self.__learning_rate * middle_error

    def __calculation_k_1(self):
        """
        Метод расчета первого коэффициента
        :return: Расчитанный первый коэффициент
        """
        result = 0
        for key in self.__traning_data:
            result += self.__error(key, self.__traning_data[key]) * key
        middle_error = result / self.__traning_data.__len__()
        return self.__learning_rate * middle_error

    def traning_model(self):
        for i in range(10):
            for _ in self.__traning_data:
                tmp_k_0 = self.k_0 - self.__calculation_k_0()
                tmp_k_1 = self.k_1 - self.__calculation_k_1()
                print(tmp_k_0, tmp_k_1)
                self.k_0 = tmp_k_0
                self.k_1 = tmp_k_1
        return self.k_0, self.k_1

