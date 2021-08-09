import parse_csv
from model_training import traning_model, estimate_price_traning

if __name__ == '__main__':
    parse_csv.parse_csv()
    result = traning_model(parse_csv.traning_data)
    print(estimate_price_traning(result[0], result[1], 240000))
