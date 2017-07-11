import pandas as pd
import re

data_path = '../data/'


def full_output(dir_prefix):

    output = pd.read_csv(dir_prefix + 'output/output'+str(0)+'.csv')

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'output/output'+str(i)+'.csv')
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_output(dir_prefix):
    output = pd.read_csv(dir_prefix + 'output/output'+str(0)+'.csv')
    output = output[output.iplace == 1]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'output/output'+str(i)+'.csv')
        read = read[read.iplace == 1]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_no_bio_output(dir_prefix):
    output = pd.read_csv(dir_prefix + 'output/output'+str(0)+'.csv')
    output = output[(output.iplace == 1) & (output.ibio == 0)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'output/output'+str(i)+'.csv')
        read = read[(read.iplace == 1) & (read.ibio == 0)]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_bio_output(dir_prefix):
    output = pd.read_csv(dir_prefix + 'output/output'+str(0)+'.csv')
    output = output[(output.iplace == 1) & (output.ibio == 1)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + 'output/output'+str(i)+'.csv')
        read = read[(read.iplace == 1) & (read.ibio == 1)]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_into_people_dic(matrix, dic1):
    dic2 = dic1
    i = 0
    while i < len(matrix):
        if matrix['idn'][i] != '':
            try:
                if 'orte' in dic2[matrix['idn'][i]].keys():
                    dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                else:
                    dic2[matrix['idn'][i]]['orte'] = {}
                    dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
            except KeyError:
                dic2[matrix['idn'][i]] = {}
                dic2[matrix['idn'][i]]['idn'] = matrix['idn'][i]
                dic2[matrix['idn'][i]]['name'] = matrix['name'][i]
                if 'orte' in dic2[matrix['idn'][i]].keys():
                    dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                else:
                    dic2[matrix['idn'][i]]['orte'] = {}
                    dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
        i = i+1

    return dic2


def location_list(matrix):
    loc = {}
    i = 0
    while i < len(matrix):
        if matrix['cities'][i] != '' and matrix['cities'][i] not in loc.keys():
            loc[matrix['cities'][i]] = 1
        else:
            loc[matrix['cities'][i]] = loc[matrix['cities'][i]] + 1
        i = i+1

    return loc

def clean_city(string):
    string1 = re.sub(' \(([^)]+)\)', '', string)
    string2 = re.sub('\(([^)]+)\)', '', string1)
    string3 = re.sub(' \<([^)]+)\>', '', string2)
    string4 = re.sub('\<([^)]+)\>', '', string3)
    string5 = re.sub(' \[([^)]+)\]', '', string4)
    string6 = re.sub('\[([^)]+)\]', '', string5)
    string7 = re.sub('\(([^)]+)', '', string6)
    string8 = re.sub('\(', '', string7)
    string9 = re.sub('\)', '', string8)
    string10 = re.sub('\?', '', string9)

    return string10


def are_clean_cities_in_bios(dic, key):
    dummies = []
    for l in dic[key]['orte'].values():
        if re.search(clean_city(l), dic[key]['leben']):
            dummies.append(1)
        else:
            print(l)
            dummies.append(0)

    return dummies

# print(len(location_list(orte_bio_output(data_path))))

#d=location_list(orte_output(data_path))
#d_list = []

#for key, value in d.items():
    #    temp = [key, value]
#    d_list.append(temp)

#d_list.sort(key=lambda x: -x[1])

#for k in range(0, 99):
#   print(d_list[k])


# print(len(locationList(orte_output(data_path))))






