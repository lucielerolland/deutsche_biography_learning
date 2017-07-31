# import extract_input as ei
# import extract_output as eo
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

data_path = '../data/'


def cut_into_sentences(paragraph):
    cut = paragraph.split('.')

    return cut


def city_match_sentence(biography, clean_city):
    sentence = []
    long_clean_city = []
    cut_biography = cut_into_sentences(biography)
    for k in cut_biography:
        if re.search(clean_city, k):
            sentence.append(k)
            long_clean_city.append(clean_city)

    return long_clean_city, sentence


def is_city_in_orte(clean_city, orte):
    if clean_city in orte:
        city_in_orte_dummy = 1
    else:
        city_in_orte_dummy = 0

    return city_in_orte_dummy

word_vect = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)


def train_sentence_to_matrix(train_sentences):
    train_features = word_vect.fit_transform(train_sentences)

    return train_features.toarray()


def target_sentence_to_matrix(target_sentences):
    target_features = word_vect.transform(target_sentences)

    return target_features.toarray()


def centered_reduced(matrix):
    mmean = np.mean(matrix, axis=0, keepdims=True)
    mstd = np.std(matrix, axis=0, keepdims=True)
    cr = (matrix-mmean)/mstd

    return cr
