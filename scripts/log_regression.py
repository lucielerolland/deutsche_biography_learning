import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(x, beta, y, l):
    m = np.shape(x)[0]
    direct_cost_vect = -y.dot(sigmoid(x.dot(beta))) - (1-y).dot(1-sigmoid(x.dot(beta)))
    reg = (l/(2*m))*np.square(beta)
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

    print(np.shape(grad))

    return grad


def gradient_update(alpha, x, beta, y, l, has_constant=True):
    beta_update = beta - alpha * gradient(x, beta, y, l, has_constant)

    return beta_update
