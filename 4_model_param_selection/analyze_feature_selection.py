from const import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY',
    'FP2.ForY',
    'FP1.ForZ',
    'FP2.ForZ',
    # 'FP1.CopX',
    # 'FP2.CopX',
    # 'FP1.CopY',
    # 'FP2.CopY'
]
feature_names = ALL_ACC_GYR_NAMES
score_funcs = ['f_regression', 'GradientBoostingRegressor', 'RandomForest']

all_result = pd.read_csv(RESULT_PATH + 'importance_matrix\\feature_selection.csv')

for i_output in range(len(output_names)):
    plt.figure()
    output_result = all_result[all_result['output_name'] == output_names[i_output]]
    for func in score_funcs:
        func_result = output_result[output_result['score_func'] == func]
        scores = func_result[feature_names].as_matrix()
        plt.plot(range(scores.shape[1]), scores[0, :], label=func)
    plt.xticks(range(scores.shape[1]))
    plt.xlabel('feature id')
    plt.ylabel('feature score')
    plt.title(output_names[i_output])
    plt.legend()
# plt.show()

select_num = 10
# select several best features
for i_output in range(len(output_names)):
    output_result = all_result[all_result['output_name'] == output_names[i_output]]
    ranks = np.zeros([len(feature_names)])
    for func in score_funcs:
        func_result = output_result[output_result['score_func'] == func]
        scores = func_result[feature_names].as_matrix()[0, :]
        sequence = np.argsort(scores)
        ranks_func = np.zeros([len(feature_names)])
        ranks_func[sequence] = np.arange(len(scores))
        ranks += ranks_func

    selected_index = np.argsort(ranks)[-select_num:].astype('int')
    selected_index = np.flip(selected_index, axis=0)
    selected_names = []
    for indice in selected_index:
        selected_names.extend([feature_names[indice]])
    print('output: ' + output_names[i_output] + ', ' + str(selected_names))



















