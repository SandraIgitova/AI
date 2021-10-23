import csv

def csv_reader(file):
    file = open(file)
    reader = csv.reader(file)
    data = []
    for line in reader:
        data.append(line)
    return data


def pars_data(data):
    new_data = []
    count = 0
    for i in range(1, 3000):
        if count < 500:
            if data[i][8] == 'False':
                new_data.append(data[i][1:9])
                count += 1
        else:
            if data[i][8] == 'True':
                new_data.append(data[i][1:9])
                count += 1
        if count > 1000:
            break

    for i in range(len(new_data)):
        new_data[i][0:7] = list(map(lambda x: float(x), new_data[i][0:7]))

    return new_data