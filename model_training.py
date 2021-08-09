def estimate_price_traning(koef_0, koef_1, millage):
    return koef_0 + koef_1 * millage


def sub_calculation_first_koefficient(millage, price, koef_0, koef_1):
    return estimate_price_traning(koef_0, koef_1, millage) - price


def sub_calculation_second_koefficient(millage, price, koef_0, koef_1):
    return (estimate_price_traning(koef_0, koef_1, millage) - price) * millage


def first_koefficient(traning_sample: dict, learning_rate: float, sample_size: int, koef_0, koef_1):
    summ = 0
    for key in traning_sample:
        summ += sub_calculation_first_koefficient(key, traning_sample[key], koef_0, koef_1)
    return learning_rate * (1 / sample_size) * summ


def second_koefficient(traning_sample: dict, learning_rate: float, sample_size: int, millage, price, koef_0, koef_1):
    summ = 0
    for key in traning_sample:
        summ += sub_calculation_second_koefficient(key, traning_sample[key], koef_0, koef_1)
    return learning_rate * (1 / sample_size) * summ


def traning_model(traning_sample: dict):
    tmp_koef_0 = 0
    tmp_koef_1 = 0
    for key in traning_sample:
        tmp_koef_0 = first_koefficient(traning_sample, 0.01, traning_sample.__len__(), key, traning_sample[key], tmp_koef_0, tmp_koef_1)
        tmp_koef_1 = second_koefficient(traning_sample, 0.01, traning_sample.__len__(), key, traning_sample[key], tmp_koef_0, tmp_koef_1)
    return tmp_koef_0, tmp_koef_1
