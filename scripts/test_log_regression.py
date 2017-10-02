import pytest
import log_regression as lr
import numpy as np


@pytest.yield_fixture()
def ones(dim_one):              # dimensions should be a 2-d tuple
    ret_ones = np.ones(dim_one)
    yield ret_ones


@pytest.yield_fixture()
def zeros(dim_zero):
    ret_zeros = np.zeros(dim_zero)
    yield ret_zeros


@pytest.yield_fixture()
def random(dim_random):
    ret_random = np.random.randn(dim_random[0], dim_random[1])
    yield ret_random


@pytest.yield_fixture()
def random_one_infinity(dim_random_infty):
    ret_random_one_infinity = np.random.randn(dim_random_infty[0], dim_random_infty[1])
    ret_random_one_infinity[0,0] = 40
    yield ret_random_one_infinity


@pytest.yield_fixture()
def t_test(size_t, ones_vec):   # y should be a matrix, ones should be a list of positions for ones encoded as tuples
    t = np.zeros(size_t)
    for k in ones_vec:
        t[k] = 1
    return t


@pytest.yield_fixture()
def beta_test_random(size_beta):   # beta should be a matrix
    beta_test = np.random.randn(size_beta[0], size_beta[1])
    return beta_test


@pytest.yield_fixture()
def beta_test_zero(size_beta):   # beta should be a matrix
    beta_test = np.zeros(size_beta)
    return beta_test


@pytest.yield_fixture()
def beta_test_identity(size_beta):   # beta should be a matrix
    beta_test = np.eye(size_beta[0], size_beta[1])
    return beta_test


@pytest.yield_fixture()
def x_test_random(size_x):   # x should be a matrix
    x_test = np.random.randn(size_x[0], size_x[1])
    return x_test


@pytest.yield_fixture()
def l():
    return np.random.randn(1, 1)[0,0]


def test_sigmoid_scalar():
    assert lr.sigmoid(0) == 1/2
    assert round(lr.sigmoid(-10)) == 0
    assert round(lr.sigmoid(10)) == 1


@pytest.mark.parametrize('dim_one', [(10, 1)])
@pytest.mark.parametrize('dim_zero', [(10, 1)])
def test_sigmoid_vector(ones, zeros):
    for i in range(10):
        assert lr.sigmoid(zeros)[i] == ones[i]*1/2
        assert round(lr.sigmoid(ones*-10)[i, 0], 3) == 0
        assert round(lr.sigmoid(ones*10)[i, 0] - ones[i, 0], 3) == 0


@pytest.mark.parametrize('dim_one', [(10, 1)])
@pytest.mark.parametrize('dim_zero', [(10, 1)])
@pytest.mark.parametrize('dim_random', [(10, 1)])
@pytest.mark.parametrize('dim_random_infty', [(10, 1)])
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


@pytest.mark.parametrize('dim_one', [(10, 6)])
@pytest.mark.parametrize('dim_zero', [(10, 6)])
@pytest.mark.parametrize('dim_random', [(10, 6)])
@pytest.mark.parametrize('dim_random_infty', [(10, 6)])
def test_softmax_matrix(ones, zeros, random, random_one_infinity):
    random_t = random.T
    assert lr.softmax(random).shape[0] == random.shape[0]
    assert lr.softmax(random).shape[1] == random.shape[1]
    assert lr.softmax(random_t).shape[0] == random_t.shape[0]
    assert lr.softmax(random_t).shape[1] == random_t.shape[1]
    assert round(lr.softmax(random_one_infinity)[0, 0]-1, 3) == 0
    assert round(lr.softmax(random_one_infinity.T)[0, 0]-1, 3) == 0
    for i in range(10):
        assert round(np.sum(lr.softmax(random), axis=1)[i]-1, 5) == 0
        for j in range(6):
            assert lr.softmax(zeros)[i, j] == ones[i, j]*(1/6)
            assert lr.softmax(zeros.T)[j, i] == ones.T[j, i]*1/10
    for j in range(6):
        assert round(np.sum(lr.softmax(random_t), axis=1)[j]-1, 5) == 0


@pytest.mark.parametrize('size_t', [(10, 5)])
@pytest.mark.parametrize('ones_vec', [[(0, 1), (1, 4), (2, 2), (3, 1), (4, 3), (5, 1), (6, 4), (7, 2), (8, 1), (9, 3)]])
@pytest.mark.parametrize('size_beta', [(5, 5)])
def test_cost_softmax_equality(t_test, beta_test_identity, l):
    assert round(lr.cost(t_test*40, beta_test_identity, t_test, 0, 'softmax', has_constant=True) + 1, 5) == 0
    reg = (l * 5) / (2 * 10)
    assert round(lr.cost(t_test*40, beta_test_identity, t_test, l, 'softmax', has_constant=False) + 1-reg, 5) == 0
    reg = (l * 4) / (2 * 10)
    assert round(lr.cost(t_test*40, beta_test_identity, t_test, l, 'softmax', has_constant=True) + 1-reg, 5) == 0


@pytest.mark.parametrize('size_t', [(10, 5)])
@pytest.mark.parametrize('ones_vec', [[]])
@pytest.mark.parametrize('size_x', [(10, 7)])
@pytest.mark.parametrize('size_beta', [(7, 5)])
def test_cost_softmax_zeros(x_test_random, t_test, beta_test_identity, l):
    assert round(lr.cost(x_test_random, beta_test_identity, t_test, 0, 'softmax', has_constant=True), 5) == 0
    reg1 = (l * 5) / (2 * 10)
    assert round(lr.cost(x_test_random, beta_test_identity, t_test, l, 'softmax', has_constant=False) - reg1, 5) == 0
    reg2 = (l * 4) / (2 * 10)
    assert round(lr.cost(x_test_random, beta_test_identity, t_test, l, 'softmax', has_constant=True) - reg2, 5) == 0


@pytest.mark.parametrize('size_t', [(10, 5)])
@pytest.mark.parametrize('ones_vec', [[(0, 1), (1, 4), (2, 2), (3, 1), (4, 3), (5, 1), (6, 4), (7, 2), (8, 1), (9, 3)]])
@pytest.mark.parametrize('size_x', [(10, 7)])
@pytest.mark.parametrize('size_beta', [(7, 5)])
def test_cost_softmax_inequality(t_test, x_test_random, beta_test_random, l):
    beta_test_first_element_pos = beta_test_random
    beta_test_first_element_pos[0, 0] = 20
    x_test_first_column_pos = x_test_random
    x_test_first_column_pos[:, 0] = 20
    assert round(lr.cost(x_test_first_column_pos, beta_test_first_element_pos, t_test, 0, 'softmax', has_constant=True), 5) == 0
    reg1 = l / (2 * 10) * np.sum(np.square(beta_test_first_element_pos))
    assert round(lr.cost(x_test_first_column_pos, beta_test_first_element_pos, t_test, l, 'softmax', has_constant=False) -reg1, 5) == 0
    reg2 = l / (2 * 10) * (np.sum(np.square(beta_test_first_element_pos)) - np.sum(np.square(beta_test_first_element_pos[:, 0])))
    assert round(lr.cost(x_test_first_column_pos, beta_test_first_element_pos, t_test, l, 'softmax', has_constant=True) -reg2, 5) == 0


@pytest.mark.parametrize('size_t', [(10, 5)])
@pytest.mark.parametrize('ones_vec', [[(0, 1), (1, 4), (2, 2), (3, 1), (4, 3), (5, 1), (6, 4), (7, 2), (8, 1), (9, 3)]])
@pytest.mark.parametrize('size_beta', [(5, 5)])
def test_gradient_softmax_equality(t_test, beta_test_identity):
    for i in range(beta_test_identity.shape[0]):
        for j in range(beta_test_identity.shape[1]):
            assert round(lr.gradient(t_test*40, beta_test_identity, t_test, 0, 'softmax', has_constant=True)[i,j], 5) == 0


@pytest.mark.parametrize('dim_random', [(10, 1)])
def test_grad_mat_to_vector_vector(random):
    for i in range(10):
        assert lr.grad_mat_to_vector(random)[i] == random[i]


@pytest.mark.parametrize('dim_random', [(10, 7)])
def test_grad_mat_to_vector_vector(random):
    for i in range(10):
        for j in range(7):
            assert lr.grad_mat_to_vector(random)[j+i*7] == random[i, j]