import matplotlib.pyplot as plt
import numpy as np

from parse_csv import parse_csv


def visualize(k0, k1, ref_k0, ref_k1, lr, model):
    # Парсинг всех точек датасета
    data_for_predict, verification_data = parse_csv()

    fig = plt.figure()
    ax = plt.axes()

    plt.grid(b='true')

    # Отображения точек из датасета
    ax.scatter(data_for_predict, verification_data, color='g')

    # Иментование осей
    ax.set_xlabel('millage')
    ax.set_ylabel('prise')

    # Используемые коэффициенты
    print('K0={},     K1={}, y = {}X + {}'.format(k0, k1, k1, k0))
    print('ref_K0={}, ref_K1={}, y = {}X + {}'.format(ref_k0, ref_k1, ref_k1, ref_k0))

    index = np.max(data_for_predict)
    a = float(index)
    b = float(lr.predict(a))

    index = np.min(data_for_predict)
    c = float(index)
    d = float(lr.predict(c))

    # Отрисовка моей линии
    plt.axline([c, d], [a, b], color='y', label='ltheresi liner regression')
    #
    # # Отрисовка линии полученной с помощью sklearn
    plt.axline((0, ref_k0), slope=ref_k1, color='b', label='sklearn liner regression', linestyle='dashed')

    # Отображение легенды
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center')

    # Отображения холста
    plt.show()


if __name__ == '__main__':
    pass
    # sklern z-standard
    # x_standard_dataset = StandardScaler().fit_transform(x_dataset)

    # Обучение модели и получение весов с помощью sklearn
    # model = LinearRegression()
    # model.fit(x_dataset, y_dataset)

    # print('my predict = {}'.format(lr.predict(x_dataset)))
    # print('sklearn predict = {}'.format(model.predict(x_dataset)))
    # arguments = len(sys.argv) - 1
    # print("The script is called with %i arguments" % (arguments))
    # visualize()
