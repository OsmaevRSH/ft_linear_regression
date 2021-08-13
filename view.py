import matplotlib.pyplot as plt
from parse_csv import parse_csv


def my_print(k0, k1, ref_k0, ref_k1):
    # Парсинг всех точек датасета
    data_for_predict, verification_data = parse_csv()

    fig = plt.figure()
    ax = plt.axes()

    plt.grid(b='true')

    # Отображения точек из датасета
    ax.scatter(data_for_predict, verification_data, color='r')

    # Иментование осей
    ax.set_xlabel('millage')
    ax.set_ylabel('prise')

    # Используемые коэффициенты
    print('K0={},     K1={}'.format(k0, k1))
    print('ref_K0={}, ref_K1={}'.format(ref_k0, ref_k1))

    # Отрисовка моей линии
    plt.axline((0, k0), slope=k1, color='y', label='ltheresi liner regression')

    # Отрисовка линии полученной с помощью sklearn
    plt.axline((0, ref_k0), slope=ref_k1, color='b', label='sklearn liner regression', linestyle='dashed')

    # Отображение легенды
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center')

    # Отображения холста
    plt.show()
