import estimation as es

data_path = '../data'
city_source = 'db'
should_rebuild = True
should_consider_all = 'full'
activation = 'sigmoid'
build_city_pred = True


iter_list = [1000]
# iterations_list = [100, 300, 1000, 3000, 10000, 30000, 100000]

a_list = [0.1]
# alpha_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

lambda_list = [0.001]
# l_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

es.estimation(activation=activation, path=data_path, source=city_source, rebuild=should_rebuild,
              subset=should_consider_all, iterations_list=iter_list, alpha_list=a_list, l_list=lambda_list,
              output=build_city_pred)
