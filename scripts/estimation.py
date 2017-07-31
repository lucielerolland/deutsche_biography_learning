import extract_output as eo
import extract_input as ei
import city_bio_matching as cbm
import log_regression as lr
import numpy as np
import random

# Import input & output

data_path = '../data/'

biographies = ei.bio_orte_input(data_path)

locations = eo.orte_bio_output(data_path)

dic_input = ei.bio_into_people_dic(biographies, dic1={})

full_dic = eo.orte_into_people_dic(locations, dic1=dic_input)

people = full_dic.keys()

ref_cities_dic = eo.location_list(locations)

ref_cities_unclean = ref_cities_dic.keys()

ref_cities_clean = []

for k in ref_cities_unclean:
    clean = eo.clean_city(k)
    if clean not in ref_cities_clean:
        ref_cities_clean.append(clean)

# Build y

y = []
city_list = []
sentence = []

# for k in people:
for k in ['136810942', '139526781', '129102687', '138361193', '116119160', '119108445', '118925563']:
        for c0 in ref_cities_clean:
            add_city_list, add_sentence = cbm.city_match_sentence(full_dic[k]['leben'], c0)
            city_list += add_city_list
            sentence += add_sentence
            for c1 in add_city_list:
                if c1 in full_dic[k]['orte']:
                    y.append(1)
                else:
                    y.append(0)

y = np.transpose(np.mat(y))

# Build features

x_no_intercept = cbm.train_sentence_to_matrix(sentence)

x_no_intercept_cr = cbm.centered_reduced(x_no_intercept)

intercept = np.transpose(np.mat(np.ones(len(sentence))))

x = np.concatenate((intercept, x_no_intercept_cr), axis=1)

# Separate train & test

n_test = round(np.shape(y)[0]*0.2)

test_sample = random.sample(range(0, len(y)), n_test)

train_sample = []

for k in range(len(y)):
    if k not in test_sample:
        train_sample.append(k)

test_y = y[test_sample]
train_y = y[train_sample]

test_x = x[test_sample]
train_x = x[train_sample]

# Gradient descent

iterations = 15000
alpha = 0.1
l = 0.1

beta = np.transpose(np.mat(np.random.randn(np.shape(x)[1])))

cost = []
index = []

for i in range(iterations):
    beta = lr.gradient_update(alpha=alpha, x=train_x, beta=beta, y=train_y, l=l, has_constant=True)
    cost.append(lr.cost(train_x, beta, train_y, l))

# Compute perf on test

test_pred_y = lr.pred(test_x, beta)

exact_pred = 0
total = 0

for i in range(test_pred_y):
    total += 1
    if test_pred_y[i] == y[i]:
        exact_pred =+ 1

print(exact_pred/total)