import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(x, beta, y, l):
    m = np.shape(x)[0]
    yhat = sigmoid(x.dot(beta))
    direct_cost_vect = -np.multiply(y, yhat) - np.multiply((1-y), (1-yhat))
    reg = (l/(2*m))*np.transpose(beta).dot(beta)
    full_cost = 1/m*np.sum(direct_cost_vect) + reg

    return full_cost


def gradient(x, beta, y, l, has_constant=True):
    m = np.shape(x)[0]
    if has_constant:
        k = np.shape(x)[1]
        mult = np.identity(k, dtype=float)
        mult[0] = 0
        reg_gradient = (l/m)*mult.dot(beta)
    else:
        reg_gradient = (l/m)*beta
    direct_gradient = (1/m)*np.transpose(x).dot(sigmoid(x.dot(beta))-y)
    grad = direct_gradient + np.mat(reg_gradient)

    return grad


def gradient_update(alpha, x, beta, y, l, has_constant=True):
    beta_update = beta - alpha * gradient(x, beta, y, l, has_constant)

    return beta_update


def pred(x, beta):
    z = sigmoid(x.dot(beta))
    pred_y = np.around(z)
    pred_y = pred_y.astype('int')

    return pred_y
