import numpy as np
import pandas as pd


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
        reg = (l/(2*m))*np.dot(beta.T, beta)
        full_cost = 1/m*np.sum(direct_cost_vect) + reg

    elif activation == 'softmax':

        y_hat = softmax(x.dot(beta))
        direct_cost_vect = np.sum(-np.multiply(y, np.log(y_hat)))
        reg = (l / 2) * np.multiply(beta, beta)
        if has_constant:
            reg[:, 0] = 0
        reg = np.sum(reg)
        full_cost = (1 / m) * (direct_cost_vect + reg)

#    if pd.isnull(full_cost):
#        for i in range(y.shape[0]):
#            print(y_hat[i], y[i])

    return full_cost


def gradient(x, beta, y, l, activation, has_constant=True):
    m = np.shape(x)[0]
    k = np.shape(x)[1]
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


def pred(x, beta, activation):
    if activation == 'softmax':
        z = softmax(x.dot(beta))
        pred_y = z.argmax(axis=1)

    elif activation == 'sigmoid':
        z = sigmoid(x.dot(beta))
        pred_y = (z > 0.5)

    return pred_y


def grad_mat_to_vector(matrix):
    vector = matrix.reshape(matrix.shape[0]*matrix.shape[1], 1)

    return vector


def grad_vec_to_mat(vector, shape):
    matrix = vector.reshape(shape)

    return matrix


def gradient_checker(x, y, beta, epsilon, l, activation, has_constant=True):

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


def build_city_list_pred(features, theta, activation, scholars, extracted_cities, is_a_living_city, true_cities_full, path):
    pred_y_full = pred(features, theta, activation)

    city_list_pred = {}

    for i in range(len(scholars)):
        if scholars[i] == 0 or (scholars[i] != 0 and scholars[i] != scholars[i-1]):
            city_list_pred[scholars[i]] = {}
            city_list_pred[scholars[i]]['rejected'] = []
            city_list_pred[scholars[i]]['lived_pred'] = []
            city_list_pred[scholars[i]]['lived_y'] = []
            if activation == 'softmax':
                city_list_pred[scholars[i]]['geburt'] = []
                city_list_pred[scholars[i]]['tod'] = []
        if pred_y_full[i] == 0:
            city_list_pred[scholars[i]]['rejected'].append(extracted_cities[i])
        elif pred_y_full[i] == 1:
            city_list_pred[scholars[i]]['lived_pred'].append(extracted_cities[i])
        elif pred_y_full[i] == 2 and activation == 'softmax':
            city_list_pred[scholars[i]]['geburt'].append(extracted_cities[i])
        elif pred_y_full[i] == 3 and activation == 'softmax':
            city_list_pred[scholars[i]]['tod'].append(extracted_cities[i])
        if is_a_living_city[i] == 1:
            city_list_pred[scholars[i]]['lived_y'].append(extracted_cities[i])

    idn_df = []
    lived_pred_df = []
    lived_y_df = []
    rejected_df = []
    true_lived_df = []
    column_names = ['idn', 'rejected', 'lived_pred', 'lived_y', 'true_lived']
    if activation == 'softmax':
        geburt_df = []
        tod_df = []
        column_names.append('geburt')
        column_names.append('tod')

    for k in set(scholars):
        idn_df.append(k)
        rejected_df.append(city_list_pred[k]['rejected'])
        lived_pred_df.append(city_list_pred[k]['lived_pred'])
        lived_y_df.append(city_list_pred[k]['lived_y'])
        if activation == 'sigmoid':
            true_lived_df.append(true_cities_full[k])
        if activation == 'softmax':
            geburt_df.append(city_list_pred[k]['geburt'])
            tod_df.append(city_list_pred[k]['tod'])

    full_df = {'idn': idn_df, 'lived_pred': lived_pred_df, 'lived_y': lived_y_df, 'true_lived': true_lived_df,
               'rejected': rejected_df}
    if activation == 'softmax':
        full_df['geburt'] = geburt_df
        full_df['tod'] = tod_df

    pd.DataFrame(full_df).to_csv(path + '/final/city_list_pred_' + activation + '.csv', encoding='utf-8', index=False, columns=column_names)