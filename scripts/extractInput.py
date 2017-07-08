import pandas as pd

data_path = '../data/'


def full_input(dir_prefix):

    input_csv =pd.read_csv(dir_prefix + 'input/input'+str(0)+'.csv')

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'input/input'+str(i)+'.csv')
        input_csv = input_csv.append(read, ignore_index=True)
        i = i+1

    return input_csv


def bio_input(dir_prefix):
    input_csv = pd.read_csv(dir_prefix + 'input/input'+str(0)+'.csv')
    input_csv = input_csv[input_csv.iplace == 1]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'input/input'+str(i)+'.csv')
        read = read[read.ibio == 1]
        input_csv = input_csv.append(read, ignore_index=True)
        i = i+1

    return input_csv


def bio_no_orte_input(dir_prefix):
    input_csv = pd.read_csv(dir_prefix + 'input/input'+str(0)+'.csv')
    input_csv = input_csv[(input_csv.ibio == 1) & (input_csv.iplace == 0)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'input/input'+str(i)+'.csv')
        read = read[(read.ibio == 1) & (read.iplace == 0)]
        input_csv = input_csv.append(read, ignore_index=True)
        i = i+1

    return input_csv


def bio_orte_input(dir_prefix):
    input_csv = pd.read_csv(dir_prefix + 'input/input'+str(0)+'.csv')
    input_csv = input_csv[(input_csv.ibio == 1) & (input_csv.iplace == 1)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'input/input'+str(i)+'.csv')
        read = read[(read.ibio == 1) & (read.iplace == 1)]
        input_csv = input_csv.append(read, ignore_index=True)
        i = i+1

    return input_csv


def bio_into_people_dic(matrix, dic1):

    dic2 = dic1
    i = 0
    while i < len(matrix):
        if matrix['idn'][i] != '':
            try:
                dic2[matrix['idn'][i]]['leben'] = matrix['leben'][i]
            except KeyError:
                dic2[matrix['idn'][i]] = {}
                dic2[matrix['idn'][i]]['idn'] = matrix['idn'][i]
                dic2[matrix['idn'][i]]['name'] = matrix['name'][i]
                dic2[matrix['idn'][i]]['leben'] = matrix['leben'][i]
        i = i+1
    return dic2
