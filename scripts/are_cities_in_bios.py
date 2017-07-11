import extract_input as ei
import extract_output as eo
import re

data_path = '../data/'


def build_people_dic(dir_prefix):
    dic = {}
    dic1 = ei.bio_into_people_dic(ei.bio_orte_input(dir_prefix), dic)
    dic2 = eo.orte_into_people_dic(eo.orte_bio_output(dir_prefix), dic1)
    return dic2


#def are_cities_in_bios(dic, key):
    #    dummies = []
    #    for l in dic[key]['orte'].values():
    #        try:
    #            if re.search(l, dic[key]['leben']):
    #                dummies.append(1)
    #            else:
    #                dummies.append(0)
    #        except:
    #            print(l, dic[key]['leben'])

#    return dummies


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
        try:
            if re.search(clean_city(l), dic[key]['leben']):
                dummies.append(1)
            else:
                print(l)
                dummies.append(0)
        except:
            print(clean_city(l))

    return dummies

test = []
testdic = build_people_dic(data_path)

for k in testdic.keys():
    test = test + are_clean_cities_in_bios(testdic, k)

print(sum(test)/len(test))

#print(clean_city('Wien (?)'))