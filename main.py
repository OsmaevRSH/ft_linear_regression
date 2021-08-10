import model_training
import parse_csv
from view import my_print

if __name__ == '__main__':
    parse_csv.parse_csv()
    lr = model_training.LinearRegression(parse_csv.traning_data)
    data = lr.traning_model()
    my_print(data[0], data[1])

