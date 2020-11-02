import json


def load_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(path):
    f = open(path, encoding='utf-8')
    return [eval(line.strip()) for line in f]
