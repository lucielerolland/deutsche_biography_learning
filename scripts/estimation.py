import extract_output as eo
import extract_input as ei
import city_bio_matching as cbm
import log_regression as lr
import numpy as np
import random
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer


def build_x_and_y(path, subset, source, activation):

    biographies = ei.bio_orte_input(path)

    locations = eo.orte_bio_output(path)

    dic_input = ei.bio_into_people_dic(biographies, dic1={})

    full_dic = eo.add_all_orte(eo.orte_into_people_dic(locations, dic1=dic_input))

    eo.save_true_scholar_city_dic(full_dic, activation, path)

    people = list(full_dic.keys())

    ref_cities_clean = eo.location_list_clean(source, locations, path)

    print('Number of unique locations', len(set(ref_cities_clean)))

    # Build y

    is_a_living_city = []
    city_list = []
    sentence = []
    scholar = []

    true_orte_dic = {}

    counter_in_orte = 0
    counter_in_both = 0

    if subset == 'full':
        people_set = people
    elif subset == 'partial':
        people_set = ['136810942', '139526781', '129102687', '138361193', '116119160', '119108445', '118925563']

    for k in people_set:
        full_dic[k]['extracted_orte'] = []
        for c0 in set(ref_cities_clean):
            is_a_city_match, add_sentence = cbm.city_match_sentence(full_dic[k]['leben'], c0)
            if is_a_city_match == 1:
                city_list.append(c0)
                full_dic[k]['extracted_orte'].append(c0)
                scholar.append(k)
                if activation == 'softmax':
                    if ('tod' in full_dic[k]['orte'].keys() and c0 == full_dic[k]['orte']['tod']) or \
                            ('grab' in full_dic[k]['clean_orte'].keys() and c0 == full_dic[k]['clean_orte']['grab']):
                        is_a_living_city.append(3)
                        sentence.append(add_sentence)
                    elif 'geburt' in full_dic[k]['clean_orte'].keys() and c0 == full_dic[k]['clean_orte']['geburt']:
                        sentence.append(add_sentence)
                        is_a_living_city.append(2)
                    elif 'wirk' in full_dic[k]['clean_orte'].keys() and c0 in full_dic[k]['clean_orte']['wirk']:
                        sentence.append(add_sentence)
                        is_a_living_city.append(1)
                    else:
                        sentence.append(add_sentence)
                        is_a_living_city.append(0)
                if activation == 'sigmoid':
                    if c0 in list(full_dic[k]['all_orte']):
                        is_a_living_city.append(1)
                        sentence.append(add_sentence)
                    else:
                        sentence.append(add_sentence)
                        is_a_living_city.append(0)
        for m in set(full_dic[k]['all_orte']):
            counter_in_orte += 1
            if m in full_dic[k]['extracted_orte']:
                counter_in_both += 1
        if activation == 'sigmoid':
            true_orte_dic[k] = full_dic[k]['all_orte']
        if activation == 'softmax':
            true_orte_dic[k] = full_dic[k]['clean_orte']

    print('Share of matched cities', counter_in_both, counter_in_orte, counter_in_both/counter_in_orte)

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

    return features, is_a_living_city, scholar, city_list

# Separate train & test


def train_and_test(features, is_a_living_city, is_a_living_city_dummies, activation):

    n_test = round(np.shape(is_a_living_city)[0]*0.2)

    test_sample = random.sample(range(0, len(is_a_living_city)), n_test)

    train_sample = []

    for k in range(len(is_a_living_city)):
        if k not in test_sample:
            train_sample.append(k)

    train, test = {}, {}

    test['y'] = is_a_living_city[test_sample]
    train['y'] = is_a_living_city[train_sample]

    test['x'] = features[test_sample]
    train['x'] = features[train_sample]

    if activation == 'softmax':
        test['y_dummies'] = is_a_living_city_dummies[test_sample]
        train['y_dummies'] = is_a_living_city_dummies[train_sample]

    return train, test

# Gradient descent


def estimation_sigmoid(path, source, rebuild, alpha_list, iterations_list, l_list, subset, output):

    if rebuild:
        print("Started building at time : " + str(datetime.now()))
        x, y, idn, cities = build_x_and_y(path, subset, source=source, activation='sigmoid')
        np.save(path + '/intermediary/x_sigmoid.npy', x)
        np.save(path + '/intermediary/y_sigmoid.npy', y)
        np.save(path + '/intermediary/idn_sigmoid.npy', idn)
        np.save(path + '/intermediary/cities_sigmoid.npy', cities)
    else:
        print("Started loading at time : " + str(datetime.now()))
        x = np.load(path + '/intermediary/x_sigmoid.npy')
        y = np.load(path + '/intermediary/y_sigmoid.npy')
        idn = np.load(path + '/intermediary/idn_sigmoid.npy')
        cities = np.load(path + '/intermediary/cities_sigmoid.npy')

    train, test = train_and_test(features=x, is_a_living_city=y, is_a_living_city_dummies=[], activation='sigmoid')

    cost = []

    print("Started training at time : " + str(datetime.now()))

    for l in l_list:
        for alpha in alpha_list:
            for iterations in iterations_list:
                beta = np.mat(np.random.randn(x.shape[1], y.shape[1])) * 0.01
                for i in range(iterations):
                    beta = lr.gradient_descent(alpha=alpha, x=train['x'], beta=beta, y=train['y'], l=l,
                                               activation='sigmoid', has_constant=True)
                    cost.append(lr.cost(train['x'], beta, train['y'], l, 'sigmoid'))

                    if i % (iterations/10) == 0:
                        print(lr.cost(train['x'], beta, train['y'], l, 'sigmoid'))

                # Compute perf on test

                pred_y_train = lr.pred(train['x'], beta, 'sigmoid')
                pred_y_test = lr.pred(test['x'], beta, 'sigmoid')

                accuracy_train, _, _, _ = lr.performance_metrics_logistic(pred_y_train, train['y'])
                accuracy_test, precision_test, recall_test, f1_test = lr.performance_metrics_logistic(pred_y_test, test['y'])

                print('At time : ' + str(datetime.now()) + ', l : ' + str(l) + ', alpha : ' + str(alpha) +
                      ', iter : ' + str(iterations) + ', score train : ' + str(accuracy_train) +
                      ', score test : ' + str(accuracy_test))

                print('Test precision : ', precision_test, ', test recall : ',
                      recall_test, ', test f1 : ', f1_test)

                if output:
                    lr.build_city_list_pred(features=x, theta=beta, activation='sigmoid', scholars=idn,
                                            extracted_cities=cities, is_a_living_city=y, path=path)


def estimation_softmax(path, source, rebuild, alpha_list, iterations_list, l_list, subset, output):
    if rebuild:
        print("Starting building at time : " + str(datetime.now()))
        x, y, idn, cities = build_x_and_y(path, subset, source=source, activation='softmax')
        np.save(path + '/intermediary/x_softmax.npy', x)
        np.save(path + '/intermediary/y_softmax.npy', y)
        np.save(path + '/intermediary/idn_softmax.npy', idn)
        np.save(path + '/intermediary/cities_softmax.npy', cities)
    else:
        print("Starting loading at time : " + str(datetime.now()))
        x = np.load(path + '/intermediary/x_softmax.npy')
        y = np.load(path + '/intermediary/y_softmax.npy')
        idn = np.load(path + '/intermediary/idn_softmax.npy')
        cities = np.load(path + '/intermediary/cities_softmax.npy')

    K = 4

    y_dummies = lr.y_to_dummies(y, K)

    train, test = train_and_test(features=x, is_a_living_city=y, is_a_living_city_dummies=y_dummies, activation='softmax')

    cost = []

    print("Starting training at time : " + str(datetime.now()))

    for l in l_list:
        for alpha in alpha_list:
            for iterations in iterations_list:
                beta = np.mat(np.random.randn(x.shape[1], y.shape[1])) * 0.01
                for i in range(iterations):
                    beta = lr.gradient_descent(alpha=alpha, x=train['x'], beta=beta, y=train['y_dummies'], l=l,
                                               activation='softmax', has_constant=True)
                    cost.append(lr.cost(train['x'], beta, train['y_dummies'], l, 'softmax'))

                    if i % (iterations / 10) == 0:
                        print(lr.cost(train['x'], beta, train['y_dummies'], l, 'softmax'))

                # Compute perf on test

                pred_y_train = lr.pred(train['x'], beta, 'softmax')
                pred_y_test = lr.pred(test['x'], beta, 'softmax')

                accuracy_train, _, _, _ = lr.performance_metrics_softmax(pred_y_train, train['y'], K)
                accuracy_test, precision_test, recall_test, f1_test = lr.performance_metrics_softmax(pred_y_test, test['y'], K)

                print('At time : ' + str(datetime.now()) + ', l : ' + str(l) + ', alpha : ' + str(alpha) +
                      ', iter : ' + str(iterations) + ', score train : ' + str(accuracy_train) +
                      ', score test : ' + str(accuracy_test))
                for k in range(K):
                    if k == 0:
                        pass
                    else:
                        print('For class ', k, ', test precision : ', precision_test[k], ', test recall : ',
                              recall_test[k], ', test f1 : ', f1_test[k])

                if output:
                    lr.build_city_list_pred(features=x, theta=beta, activation='softmax', scholars=idn,
                                            extracted_cities=cities, path=path, is_a_living_city=y)


def estimation(activation, path, source, rebuild, alpha_list, iterations_list, l_list, subset, output):
    if activation == 'softmax':
        estimation_softmax(path, source, rebuild, alpha_list, iterations_list, l_list, subset, output)
    elif activation == 'sigmoid':
        estimation_sigmoid(path, source, rebuild, alpha_list, iterations_list, l_list, subset, output)