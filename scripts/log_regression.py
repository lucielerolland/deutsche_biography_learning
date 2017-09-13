import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector/np.exp(np.sum(exp_vector, axis=1))#, keepdims=True))


def y_to_dummies(vector, K):
    dummies = np.zeros((vector.shape[0], K))
    for i in range(vector.shape[0]):
        dummies[i, vector[i]] = 1
    return dummies


def cost(x, beta, y, l, activation):
    m = np.shape(x)[0]
    if activation == 'sigmoid':
        y_hat = sigmoid(x.dot(beta))
        direct_cost_vect = -np.multiply(y, y_hat) - np.multiply((1-y), (1-y_hat))
        reg = (l/(2*m))*np.transpose(beta).dot(beta)
        full_cost = 1/m*np.sum(direct_cost_vect) + reg

    elif activation == 'softmax':
        y_hat = softmax(x.dot(beta))
        direct_cost_vect = -np.multiply(y,y_hat)
        reg = (l / (2 * m)) * np.multiply(beta, beta)
        full_cost = 1 / m * (np.sum(direct_cost_vect) + np.sum(reg))

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


def gradient_descent(alpha, x, beta, y, l, has_constant=True):
    beta_update = beta - alpha * gradient(x, beta, y, l, has_constant)

    return beta_update


def pred(x, beta):
    z = softmax(x.dot(beta))
    pred_y = z.argmax(axis=1)

    return pred_y
