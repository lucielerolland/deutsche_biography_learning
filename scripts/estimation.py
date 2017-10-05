
import extract_output as eo
import extract_input as ei
import city_bio_matching as cbm
import log_regression as lr
import numpy as np
import random
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer

# Import input & output


def build_x_and_y(path):

    biographies = ei.bio_orte_input(path)

    locations = eo.orte_bio_output(path)

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

    is_a_living_city = []
    city_list = []
    sentence = []
    scholar = []

    # for k in people:
    for k in ['136810942', '139526781', '129102687', '138361193', '116119160', '119108445', '118925563']:
        full_dic[k]['extracted_orte'] = {}
        for c0 in set(ref_cities_clean):
            is_a_city_match, add_sentence = cbm.city_match_sentence(full_dic[k]['leben'], c0)
            if is_a_city_match == 1:
                city_list.append(c0)
                scholar.append(k)
                if ('tod' in full_dic[k]['orte'].keys() and c0 == full_dic[k]['orte']['tod']) or \
                        ('grab' in full_dic[k]['clean_orte'].keys() and c0 == full_dic[k]['clean_orte']['grab']):
                    is_a_living_city.append(3)
                    sentence.append(add_sentence)
                elif 'geburt' in full_dic[k]['clean_orte'].keys() and c0 == full_dic[k]['clean_orte']['geburt']:
                    sentence.append(add_sentence)
                    is_a_living_city.append(2)
                elif 'wirk' in full_dic[k]['clean_orte'].keys() and c0 in full_dic[k]['orte']['wirk']:
                    sentence.append(add_sentence)
                    is_a_living_city.append(1)
                else:
                    sentence.append(add_sentence)
                    is_a_living_city.append(0)

    is_a_living_city = np.transpose(np.mat(is_a_living_city))

    print(len(is_a_living_city))
    print(len(sentence))
    print(len(city_list))
    print(len(scholar))

    # Build features

    word_vect = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=ref_cities_clean,
                                max_features=1000)

    features_no_intercept = cbm.train_sentence_to_matrix(sentence, word_vect)

    features_no_intercept_cr = cbm.centered_reduced(features_no_intercept)

    intercept = np.transpose(np.mat(np.ones(len(sentence))))

    features = np.concatenate((intercept, features_no_intercept_cr), axis=1)

    return features, is_a_living_city

# Separate train & test


def train_and_test(features, is_a_living_city, is_a_living_city_dummies): #, city_lists, scholars):

    n_test = round(np.shape(is_a_living_city)[0]*0.2)

    test_sample = random.sample(range(0, len(is_a_living_city)), n_test)

    train_sample = []

    for k in range(len(is_a_living_city)):
        if k not in test_sample:
            train_sample.append(k)

    test_is_a_living_city = is_a_living_city[test_sample]
    train_is_a_living_city = is_a_living_city[train_sample]

    test_features = features[test_sample]
    train_features = features[train_sample]

    test_is_a_living_city_dummies = is_a_living_city_dummies[test_sample]
    train_is_a_living_city_dummies = is_a_living_city_dummies[train_sample]

#    test_city_lists = city_lists[test_sample]
#    train_city_lists = city_lists[train_sample]

#    test_scholars = scholars[test_sample]
#    train_scholars = scholars[train_sample]

    return train_features, train_is_a_living_city, train_is_a_living_city_dummies, test_features, test_is_a_living_city\
        , test_is_a_living_city_dummies #, test_city_lists, train_city_lists, test_scholars, train_scholars

# Gradient descent

rebuild = False

if rebuild:
    print("Starting building at time : " + str(datetime.now()))
    x, y = build_x_and_y('../data/')
    np.save('x.npy', x)
    np.save('y.npy', y)
else:
    print("Starting loading at time : " + str(datetime.now()))
    x = np.load('x.npy')
    y = np.load('y.npy')

# scholar_dic = np.load('scholar_dic.npy')

# Number of classes
K = 4

y_dummies = lr.y_to_dummies(y, K)

#train_x, train_y, train_y_dummies, test_x, test_y, test_y_dummies, test_city_list, train_city_list, test_scholar\
#    , train_scholar = train_and_test(x, y, y_dummies, np.matrix(city_list).T, np.matrix(scholar).T)

train_x, train_y, train_y_dummies, test_x, test_y, test_y_dummies = train_and_test(x, y, y_dummies)

epsilon = 1e-7

# iterations_list = [1000]
iterations_list = [100, 300, 1000, 3000, 10000, 30000, 100000]

alpha_list = [0.3]
# alpha_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

l_list = [0.001]
# l_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

# beta = np.mat(np.random.randn(x.shape[1], y_dummies.shape[1]))*0.01

cost = []
index = []

print("Starting training at time : " + str(datetime.now()))

# print(lr.gradient_checker(x=train_x, y=train_y_dummies, beta=beta, epsilon=epsilon, l=l_list[0], has_constant=True))
for l in l_list:
    for alpha in alpha_list:
        for iterations in iterations_list:
            beta = np.mat(np.random.randn(x.shape[1], y_dummies.shape[1]))*0.01
            for i in range(iterations):
                beta = lr.gradient_descent(alpha=alpha, x=train_x, beta=beta, y=train_y_dummies, l=l, activation='softmax', has_constant=True)
                cost.append(lr.cost(train_x, beta, train_y_dummies, l, 'softmax'))

                if i % (iterations/5) == 0:
                    print(lr.cost(train_x, beta, train_y_dummies, l, 'softmax'))
#                    print(lr.gradient_checker(x=x, y=y, beta=beta, epsilon=epsilon, l=l_list[0], has_constant=True))

            # Compute perf on test

            pred_y_train = lr.pred(train_x, beta)
            pred_y_test = lr.pred(test_x, beta)

            exact_pred_train = 0
            exact_pred_test = 0

            total_train = 0
            total_test = 0

            for i in range(np.shape(pred_y_train)[0]):
                total_train += 1
                if pred_y_train[i] == train_y[i]:
                    exact_pred_train += 1

            for i in range(np.shape(pred_y_test)[0]):
                total_test += 1
                if pred_y_test[i] == test_y[i]:
                    exact_pred_test += 1

            print('At time : ' + str(datetime.now()) + ', l : ' + str(l) + ', alpha : ' + str(alpha) +
                  ', iter : ' + str(iterations) + ', score train : ' + str(exact_pred_train/total_train) +
                  ', score test : ' + str(exact_pred_test/total_test))
