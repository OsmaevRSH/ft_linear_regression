import matplotlib.pyplot as plt
import csv

traning_data = dict()


def my_print(k0, k1):
    fig = plt.figure()
    # plt.axis([0, 10, 0, 10])
    plt.grid(True)
    for key in traning_data:
        plt.scatter(key, traning_data[key], s=100, marker='o')
    plt.axline((-k0, 0), slope=k1, color='r')
    plt.show()


with open('data.csv', 'r', newline='') as file_data:
    reader_data = csv.reader(file_data)
    next(reader_data)
    for index in reader_data:
        traning_data.update({index[0]: index[1]})

with open('save_koef.csv', 'r', newline='') as file_data:
    reader_data = csv.reader(file_data)
    for index in reader_data:
        data = index
my_print(float(data[0]), float(data[1]))



