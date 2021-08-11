import numpy as np
from numpy import genfromtxt


def parse_csv():
    traning_set = genfromtxt('data.csv', delimiter=',')
    traning_set = np.delete(traning_set, 0, 0)
    traning_set = np.hsplit(traning_set, 2)
    data_for_predict = np.array(traning_set[0])
    verification_data = np.array(traning_set[1])
    return data_for_predict, verification_data
