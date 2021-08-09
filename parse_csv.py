import csv

traning_data = dict()


def parse_csv(filename='data.csv'):
    """
        Метод парсинга файла с выбокой, для обучения модели
    """
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for rows in reader:
            traning_data.update({int(rows[0]): int(rows[1])})
