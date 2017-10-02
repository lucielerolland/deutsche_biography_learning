import numpy as np


def sigmoid(z):

    return 1/(1+np.exp(-z))


def softmax(matrix):
    exp_matrix = np.exp(matrix)
    try:
        sum_exp_matrix = np.sum(exp_matrix, axis=1, keepdims=True)
    except :
        sum_exp_matrix = np.sum(exp_matrix, axis=1)

    return exp_matrix/sum_exp_matrix


def y_to_dummies(vector, K):
    dummies = np.zeros((vector.shape[0], K))
    for i in range(vector.shape[0]):
        dummies[i, vector[i]] = 1

    return dummies


def cost(x, beta, y, l, activation, has_constant=True):

    m = np.shape(x)[0]
    if activation == 'sigmoid':
        y_hat = sigmoid(x.dot(beta))
        direct_cost_vect = -np.multiply(y, np.log(y_hat)) - np.multiply((1-y), np.log(1-y_hat))
        reg = (l/(2*m))*np.square(beta)
        full_cost = 1/m*np.sum(direct_cost_vect) + reg

    elif activation == 'softmax':

        y_hat = softmax(x.dot(beta))
        direct_cost_vect = np.sum(-np.multiply(y, np.log(y_hat)))
        reg = (l / 2) * np.multiply(beta, beta)
        if has_constant:
            reg[:, 0] = 0
        reg = np.sum(reg)
        full_cost = (1 / m) * (direct_cost_vect + reg)

    return full_cost


def gradient(x, beta, y, l, activation, has_constant=True):
    m = np.shape(x)[0]
    k = np.shape(x)[1]
    class_num = np.shape(y)[1]
    if has_constant:
        mult = np.identity(k, dtype=float)
        mult[0, 0] = 0
        reg_gradient = (l/m)*mult.dot(beta)
    else:
        reg_gradient = (l/m)*beta
    direct_gradient = 0
    if activation == 'sigmoid':
        direct_gradient = (1 / m) * np.transpose(x).dot(sigmoid(x.dot(beta)) - y)
    elif activation == 'softmax':
        direct_gradient = (1 / (2*m)) * np.transpose(x).dot(softmax(x.dot(beta)) - y)
    grad = direct_gradient + np.mat(reg_gradient)

    return grad


def gradient_descent(alpha, x, beta, y, l, activation, has_constant=True):
    beta_update = beta - alpha * gradient(x, beta, y, l, activation, has_constant)

    return beta_update


def pred(x, beta):
    z = softmax(x.dot(beta))
    pred_y = z.argmax(axis=1)

    return pred_y


def grad_mat_to_vector(matrix):
    vector = matrix.reshape(matrix.shape[0]*matrix.shape[1], 1)

    return vector


def grad_vec_to_mat(vector, shape):
    matrix = vector.reshape(shape)

    return matrix


def gradient_checker(x, y, beta, epsilon, l, has_constant=True):

    beta_shape = (beta.shape[0], beta.shape[1])
#    beta_name = np.zeros(beta_shape)
    #    for i in beta_shape[0]:
    #    for j in beta_shape[1]:
    #        beta_name[i, j] = 'beta[' + str(i) + ', ' + str(j) + ']'
    true_grad = grad_mat_to_vector(gradient(x, beta, y, l, 'softmax', has_constant))
    vec_beta = grad_mat_to_vector(beta)
#    vec_beta_name = grad_mat_to_vector(beta_name)
    Jplus = np.zeros((len(vec_beta), 1))
    Jminus = np.zeros((len(vec_beta), 1))
    fake_grad = np.zeros((len(vec_beta), 1))
    for i in range(len(vec_beta)):
        thetaplus = vec_beta
        thetaplus[i] = thetaplus[i] + epsilon
        Jplus[i] = cost(x, grad_vec_to_mat(thetaplus, beta.shape), y, l, 'softmax')

        thetaminus = vec_beta
        thetaminus[i] = thetaminus[i] - epsilon
        Jminus[i] = cost(x, grad_vec_to_mat(thetaminus, beta.shape), y, l, 'softmax')

        fake_grad[i] = (Jplus[i] - Jminus[i])/(2*epsilon)

#        print(fake_grad[i], true_grad[i])

    diff_size = np.linalg.norm(true_grad-fake_grad)/(np.linalg.norm(true_grad) + np.linalg.norm(fake_grad))

    return 'Diff:' + str(diff_size)
