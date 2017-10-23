import pandas as pd
import re


def full_output(dir_prefix):

    output = pd.read_csv(dir_prefix + '/raw/output/output'+str(0)+'.csv')

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + '/raw/output/output'+str(i)+'.csv')
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_output(dir_prefix):
    output = pd.read_csv(dir_prefix + '/raw/output/output'+str(0)+'.csv')
    output = output[output.iplace == 1]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + '/raw/output/output'+str(i)+'.csv')
        read = read[read.iplace == 1]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_no_bio_output(dir_prefix):
    output = pd.read_csv(dir_prefix + '/raw/output/output'+str(0)+'.csv')
    output = output[(output.iplace == 1) & (output.ibio == 0)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + '/raw/output/output'+str(i)+'.csv')
        read = read[(read.iplace == 1) & (read.ibio == 0)]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def orte_bio_output(dir_prefix):
    output = pd.read_csv(dir_prefix + '/raw/output/output'+str(0)+'.csv')
    output = output[(output.iplace == 1) & (output.ibio == 1)]

    i = 1
    while i < 148:
        read = pd.read_csv(dir_prefix + '/raw/output/output'+str(i)+'.csv')
        read = read[(read.iplace == 1) & (read.ibio == 1)]
        output = output.append(read, ignore_index=True)
        i = i+1

    return output


def clean_city(string):
    string1 = re.sub(' \(([^)]+)\)', '', string)
    string2 = re.sub('\(([^)]+)\)', '', string1)
    string3 = re.sub(' <([^)]+)>', '', string2)
    string4 = re.sub('<([^)]+)>', '', string3)
    string5 = re.sub(' \[([^)]+)\]', '', string4)
    string6 = re.sub('\[([^)]+)\]', '', string5)
    string7 = re.sub('\(([^)]+)', '', string6)
    string8 = re.sub('\(', '', string7)
    string9 = re.sub('\)', '', string8)
    string10 = re.sub('\?', '', string9)

    try:
        if string10[-1] == '.':
            string10 = string10[:-1]
    except IndexError:
        pass

    try:
        if string10[-1] == ' ':
            string10 = string10[:-1]
    except IndexError:
        pass

    return string10


def orte_into_people_dic(matrix, dic1):
    dic2 = dic1
    i = 0
    while i < len(matrix):
        if matrix['idn'][i] != '' and not pd.isnull(matrix['idn'][i]):
            try:
                if re.search('wirk', matrix['funct'][i]):
                    if 'orte' not in dic2[matrix['idn'][i]].keys():
                        dic2[matrix['idn'][i]]['orte'] = {}
                        dic2[matrix['idn'][i]]['clean_orte'] = {}
                    if 'wirk' not in dic2[matrix['idn'][i]]['orte'].keys():
                        dic2[matrix['idn'][i]]['orte']['wirk'] = []
                        dic2[matrix['idn'][i]]['orte']['wirk'].append(matrix['cities'][i])
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'] = []
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'].append(clean_city(matrix['cities'][i]))
                    else:
                        dic2[matrix['idn'][i]]['orte']['wirk'].append(matrix['cities'][i])
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'].append(clean_city(matrix['cities'][i]))
                else:
                    if 'orte' not in dic2[matrix['idn'][i]].keys():
                        dic2[matrix['idn'][i]]['orte'] = {}
                        dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                        dic2[matrix['idn'][i]]['clean_orte'] = {}
                        dic2[matrix['idn'][i]]['clean_orte'][matrix['funct'][i]] = clean_city(matrix['cities'][i])
                    else:
                        dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                        dic2[matrix['idn'][i]]['clean_orte'][matrix['funct'][i]] = clean_city(matrix['cities'][i])
            except KeyError:
                dic2[matrix['idn'][i]] = {}
                dic2[matrix['idn'][i]]['idn'] = matrix['idn'][i]
                dic2[matrix['idn'][i]]['name'] = matrix['name'][i]
                if re.search('wirk', matrix['funct'][i]):
                    if 'orte' not in dic2[matrix['idn'][i]].keys():
                        dic2[matrix['idn'][i]]['orte'] = {}
                        dic2[matrix['idn'][i]]['clean_orte'] = {}
                    if 'wirk' not in dic2[matrix['idn'][i]]['orte'].keys():
                        dic2[matrix['idn'][i]]['orte']['wirk'] = []
                        dic2[matrix['idn'][i]]['orte']['wirk'].append(matrix['cities'][i])
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'] = []
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'].append(clean_city(matrix['cities'][i]))
                    else:
                        dic2[matrix['idn'][i]]['orte']['wirk'].append(matrix['cities'][i])
                        dic2[matrix['idn'][i]]['clean_orte']['wirk'].append(clean_city(matrix['cities'][i]))
                else:
                    if 'orte' not in dic2[matrix['idn'][i]].keys():
                        dic2[matrix['idn'][i]]['orte'] = {}
                        dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                        dic2[matrix['idn'][i]]['clean_orte'] = {}
                        dic2[matrix['idn'][i]]['clean_orte'][matrix['funct'][i]] = clean_city(matrix['cities'][i])
                    else:
                        dic2[matrix['idn'][i]]['orte'][matrix['funct'][i]] = matrix['cities'][i]
                        dic2[matrix['idn'][i]]['clean_orte'][matrix['funct'][i]] = clean_city(matrix['cities'][i])
        i = i+1

    return dic2


def add_all_orte(dic):
    dic1 = dic
    for k in dic1.keys():
        dic1[k]['all_orte'] = []
        for l in list(dic1[k]['clean_orte'].keys()):
            if isinstance(dic1[k]['clean_orte'][l], list):
                dic1[k]['all_orte'] += list(dic1[k]['clean_orte'][l])
            else:
                dic1[k]['all_orte'].append(dic1[k]['clean_orte'][l])
        dic1[k]['all_orte'] = set(dic1[k]['all_orte'])

    return dic1


def location_list(matrix):
    loc = {}
    i = 0
    while i < len(matrix):
        if re.search('^[a-zA-ZäöüßÄÖÜ]+', matrix['cities'][i]):
            if matrix['cities'][i] not in loc.keys():
                loc[matrix['cities'][i]] = 1
            else:
                loc[matrix['cities'][i]] = loc[matrix['cities'][i]] + 1
        i = i+1

    return loc


def clean_city_list(city_list):
    cleaned_city_list = []
    for i in range(len(city_list)):
        cleaned_city_list.append(clean_city(city_list[i]))

    return set(cleaned_city_list)


def location_list_clean(source, locations, path):
    if source == 'db':
        ref_cities_dic = location_list(locations)
        ref_cities_unclean = ref_cities_dic.keys()
        city_list = []
        for k in ref_cities_unclean:
            clean = clean_city(k)
            if clean not in city_list and clean != 'Au':
                city_list.append(clean)

    if source == 'sb':
        sb_raw = pd.read_csv(path + '/staedtebuch_city_locations.csv', encoding='latin1')
        city_list = set(sb_raw['markettown'])

    return city_list


def are_clean_cities_in_bios(dic, key):
    dummies = []
    for l in dic[key]['orte'].values():
        if re.search(clean_city(l), dic[key]['leben']):
            dummies.append(1)
        else:
            print(l)
            dummies.append(0)

    return dummies


def save_true_scholar_city_dic(dic, activation, path):

    save_path = path + '/intermediary/'

    if activation == 'sigmoid':
        scholars = []
        cities = []
        for k in dic.keys():
            scholars.append(k)
            if 'all_orte' in dic[k].keys():
                cities.append(dic[k]['all_orte'])
            else:
                cities.append('')

        pd.DataFrame({'idn': scholars, 'true_lived': cities})\
            .to_csv(save_path + 'true_scholar_dic_' + activation + '.csv', index=False,
                    columns=['idn', 'true_lived'], encoding='utf-8')
    if activation == 'softmax':
        scholars = []
        geburt = []
        wirk = []
        tod = []
        for k in dic.keys():
            scholars.append(k)
            if 'geburt' in dic[k]['clean_orte'].keys():
                geburt.append(dic[k]['clean_orte']['geburt'])
            else:
                geburt.append('')
            if 'wirk' in dic[k]['clean_orte'].keys():
                wirk.append(dic[k]['clean_orte']['wirk'])
            else:
                wirk.append('')
            if not ('tod' in dic[k]['clean_orte'].keys() or 'grab' in dic[k]['clean_orte'].keys()):
                tod.append('')
            elif 'tod' in dic[k]['clean_orte'].keys() and 'grab' in dic[k]['clean_orte'].keys():
                tod.append([dic[k]['clean_orte']['tod'], dic[k]['clean_orte']['grab']])
            elif 'tod' in dic[k]['clean_orte'].keys():
                tod.append(dic[k]['clean_orte']['tod'])
            elif 'grab' in dic[k]['clean_orte'].keys():
                tod.append(dic[k]['clean_orte']['grab'])

        pd.DataFrame({'idn': scholars, 'true_geburt': geburt, 'true_lived': wirk, 'true_tod': tod}).\
            to_csv(save_path + 'true_scholar_dic_' + activation + '.csv', index=False,
                   columns=['idn', 'true_geburt', 'true_lived', 'true_tod'], encoding='utf-8')


def load_true_scholar_city_dic(activation, path):
    load_path = path + '/intermediary/'

    true_scholar_city_dic = pd.read_csv(load_path + 'true_scholar_dic_' + activation + '.csv', encoding='utf-8')

    return true_scholar_city_dic
