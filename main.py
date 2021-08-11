import csv

import model_training
import parse_csv
from view import my_print

if __name__ == '__main__':
    data_for_predict, verification_data = parse_csv.parse_csv()
    lr = model_training.LinearRegression(data_for_predict, verification_data)
    k0, k1 = lr.traning_model()
    print(lr.predict_price(4))
    with open('save_koef.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([k0, k1])
