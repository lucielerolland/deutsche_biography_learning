import re
import numpy as np


def cut_into_sentences(paragraph):
    cut = paragraph.split('.')

    return cut


def city_match_sentence(biography, clean_city):
    sentence = ''
    city_match_indicator = 0
    cut_biography = cut_into_sentences(biography)
    for k in cut_biography:
        if re.search(clean_city, k):
            sentence += k + ' '
            city_match_indicator = 1

    return city_match_indicator, sentence


def is_city_in_orte(clean_city, clean_orte):
    if clean_city in clean_orte:
        city_in_orte_dummy = 1
    else:
        city_in_orte_dummy = 0

    return city_in_orte_dummy


def train_sentence_to_matrix(train_sentences, vectorizer):
    train_features = vectorizer.fit_transform(train_sentences)

    return train_features.toarray()


def target_sentence_to_matrix(target_sentences, vectorizer):
    target_features = vectorizer.transform(target_sentences)

    return target_features.toarray()


def centered_reduced(matrix):
    mmean = np.mean(matrix, axis=0, keepdims=True)
    mstd = np.std(matrix, axis=0, keepdims=True)
    cr = (matrix-mmean)/mstd

    return cr
