import pytest
import log_regression as lr
import numpy as np


@pytest.yield_fixture()
def ones(dimensions):              # dimensions should be a 2-d tuple
    ret_ones = np.ones(dimensions)
    yield ret_ones


@pytest.yield_fixture()
def zeros(dimensions):
    ret_zeros = np.zeros(dimensions)
    yield ret_zeros


@pytest.yield_fixture()
def random(dimensions):
    ret_random = np.random.randn(dimensions[0], dimensions[1])
    yield ret_random


@pytest.yield_fixture()
def random_one_infinity(dimensions):
    ret_random_one_infinity = np.random.randn(dimensions[0], dimensions[1])
    ret_random_one_infinity[0,0] = 40
    yield ret_random_one_infinity


def test_sigmoid_scalar():
    assert lr.sigmoid(0) == 1/2
    assert round(lr.sigmoid(-10)) == 0
    assert round(lr.sigmoid(10)) == 1


@pytest.mark.parametrize('dimensions', [(10, 1)])
def test_sigmoid_vector(ones, zeros):
    for i in range(10):
        assert lr.sigmoid(zeros)[i] == ones[i]*1/2
        assert round(lr.sigmoid(ones*-10)[i, 0], 3) == 0
        assert round(lr.sigmoid(ones*10)[i, 0] - ones[i, 0], 3) == 0


@pytest.mark.parametrize('dimensions', [(10, 1)])
def test_softmax_vector(ones, zeros, random, random_one_infinity):
    random_t = random.T
    assert lr.softmax(random).shape[0] == random.shape[0]
    assert lr.softmax(random).shape[1] == random.shape[1]
    assert lr.softmax(random_t).shape[0] == random_t.shape[0]
    assert lr.softmax(random_t).shape[1] == random_t.shape[1]
    assert round(lr.softmax(random_one_infinity.T)[0, 0]-1, 3) == 0
    for i in range(10):
        assert round(lr.softmax(zeros)[i, 0] - ones[i, 0], 3) == 0
        assert lr.softmax(zeros.T)[0, i] == ones.T[0, i]*1/10
    if i != 0:
        assert round(lr.softmax(random_one_infinity.T)[0, i], 3) == 0


@pytest.mark.parametrize('dimensions', [(10, 6)])
def test_softmax_vector(ones, zeros, random, random_one_infinity):
    random_t = random.T
    assert lr.softmax(random).shape[0] == random.shape[0]
    assert lr.softmax(random).shape[1] == random.shape[1]
    assert lr.softmax(random_t).shape[0] == random_t.shape[0]
    assert lr.softmax(random_t).shape[1] == random_t.shape[1]
    assert round(lr.softmax(random_one_infinity.T)[0, 0]-1, 3) == 0
    for i in range(10):
        for j in range(6):
            assert lr.softmax(zeros)[i, j] == ones[i, j]*(1/6)
            assert lr.softmax(zeros.T)[j, i] == ones.T[j, i]*1/10
