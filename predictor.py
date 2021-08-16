import pandas as pd


def predict(data):
    try:
        coefficients = pd.read_csv('coefficients.csv')
        k_0 = float(coefficients['k_0'])
        k_1 = float(coefficients['k_1'])
        dispersion = float(coefficients['dispersion'])
        mean = float(coefficients['mean'])
    except pd.errors.EmptyDataError:
        raise Exception('No columns to parse from file!')
    except TypeError:
        raise Exception('There are no coefficients in the file to calculate, or the data type is wrong!')
    except FileNotFoundError:
        raise Exception('File with coefficients not found, run training.py!')
    except Exception:
        raise
    else:
        return k_0 + ((data - mean) / dispersion) * k_1


if __name__ == '__main__':
    try:
        data_for_predict = float(input('Enter data for predict: '))
        predicted_value = predict(data_for_predict)
    except ValueError:
        print('You entered invalid data for predict!')
    except Exception as e:
        print(str(e))
    else:
        print('Predicted data = {}'.format(predicted_value))
