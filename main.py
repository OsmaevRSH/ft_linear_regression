import csv

import numpy as np

import model_training
import parse_csv

from sklearn.linear_model import LinearRegression

from view import my_print

if __name__ == '__main__':
    data_for_predict, verification_data = parse_csv.parse_csv()
    lr = model_training.LinearRegression(data_for_predict, verification_data)
    k0, k1 = lr.traning_model()
    print(lr.predict_price(4))

    x = np.array(data_for_predict).reshape((-1, 1))
    y = np.array(verification_data)
    model = LinearRegression().fit(x, y)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([float(k0), float(k1)])
    my_print(float(k0), float(k1))
