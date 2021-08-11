import model_training
import parse_csv

if __name__ == '__main__':
    data_for_predict, verification_data = parse_csv.parse_csv()
    lr = model_training.LinearRegression(data_for_predict, verification_data)
    k0, k1 = lr.traning_model()
    # print(k0, k1)
