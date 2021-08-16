import sys

import pandas as pd
from visualize import visualize


class Predictor:

    def __init__(self):
        self.__predicted_value = 0.0
        self.__k_0 = 0.0
        self.__k_1 = 0.0
        self.__dispersion = 0.0
        self.__mean = 0.0

    def __parse_coefficients(self):
        try:
            coefficients = pd.read_csv('coefficients.csv')
            self.__k_0 = float(coefficients['k_0'])
            self.__k_1 = float(coefficients['k_1'])
            self.__dispersion = float(coefficients['dispersion'])
            self.__mean = float(coefficients['mean'])
        except pd.errors.EmptyDataError:
            raise Exception('No columns to parse from file!')
        except TypeError:
            raise Exception('There are no coefficients in the file to calculate, or the data type is wrong!')
        except FileNotFoundError:
            raise Exception('File with coefficients not found, run training.py!')
        except Exception:
            raise

    def predict(self, data):
        try:
            self.__parse_coefficients()
            self.__predicted_value = self.__k_0 + ((data - self.__mean) / self.__dispersion) * self.__k_1
        except Exception:
            raise
        else:
            return self.__predicted_value

    def visualize_data(self):
        visualize(self.predict, value_for_predict)


if __name__ == '__main__':
    visualize_data = False

    if len(sys.argv) - 1 > 0:
        if sys.argv[1] == '--visualize':
            visualize_data = True
    try:
        pr = Predictor()
        value_for_predict = float(input('Enter value for predict: '))
        predicted_value = pr.predict(value_for_predict)
        if predicted_value != 0 and visualize_data:
            pr.visualize_data()
    except ValueError:
        print('You entered invalid data for predict!')
    except Exception as e:
        print(str(e))
    else:
        print('Predicted data = {}'.format(predicted_value))
