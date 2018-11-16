# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import PredefinedSplit
from DatabaseInfo import DatabaseInfo
from EvaluationClass import Evaluation
from SubjectData import SubjectData
from XYGenerator import XYGeneratorUni
from const import *
import numpy as np

output_names = [
    'FP1.ForX',
    # 'FP2.ForX',
    # 'FP1.ForY',
    # 'FP2.ForY',
    # 'FP1.ForZ',
    #  'FP2.ForZ',
    # 'FP1.CopX', 'FP1.CopY',
    # 'FP2.CopX',
    # 'FP2.CopY'
]
total_result_columns = ['segment', 'subject_id', 'speed', 'x_offset', 'y_offset', 'z_offset', 'theta_offset']
total_result_columns.extend(output_names)  # change TOTAL_RESULT_COLUMNS

input_names = [
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
    'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
    'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
    'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
    'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
    'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
    'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
    'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
    'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
    'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
]
my_database_info = DatabaseInfo()
total_score_df = Evaluation.initialize_result_df(total_result_columns)

train_set_num = 10000
test_set_num = 1000
test_fold = np.concatenate((np.zeros([train_set_num]), -np.ones([test_set_num])), axis=None)

ps = PredefinedSplit(test_fold=test_fold)

# evaluators = GridSearchCV(ensemble.RandomForestRegressor(n_estimators=300, max_depth=10, min_impurity_decrease=1e-5,
#                                            min_samples_split=20, max_features=0.5, n_jobs=7),
#                           param_grid={'max_depth': [5, 10, 20]}, cv=ps)
# evaluators = GridSearchCV(ensemble.GradientBoostingRegressor(n_estimators=500, learning_rate=0.1,
#                                                              min_impurity_decrease=0.001,
#                                                              min_samples_split=0.001, n_iter_no_change=5,
#                                                              validation_fraction=0.2, max_depth=6),
#                           param_grid={'tol': [1e-3, 1e-5, 1e-7]}, cv=ps)
evaluators = GridSearchCV(SVR(gamma=0.01, epsilon=0.1, C=10, max_iter=1e4), param_grid={'epsilon': [1e-1, 1e-5, 1e-7]}, cv=ps)
# evaluators = GridSearchCV(MLPRegressor(hidden_layer_sizes=(40, 8), early_stopping=True, n_iter_no_change=5, learning_rate_init=0.01, validation_fraction=0.2),
#                           param_grid={}, cv=ps)      #, tol=1e-8
# evaluators = GridSearchCV(KNeighborsRegressor(n_neighbors=8, n_jobs=7),
#                           param_grid={'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}, cv=ps)

segment_moved = SEGMENT_NAMES[0]
i_sub = 3
speed = SPEEDS[0]

all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub]
x_train = other_sub_data[input_names]
y_train = other_sub_data[output_names]
the_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
x_test = the_sub_data[input_names]
y_test = the_sub_data[output_names]

# shuffle the data
x_train, y_train = shuffle(x_train, y_train)
x_train, y_train = x_train.as_matrix()[:train_set_num, :], y_train.as_matrix()[:train_set_num, :]
x_test, y_test = shuffle(x_test, y_test)
x_test, y_test = x_test.as_matrix()[:test_set_num, :], y_test.as_matrix()[:test_set_num, :]

x_scaler = preprocessing.StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
y_scaler = preprocessing.StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

x = np.row_stack([x_train, x_test])
y = np.row_stack([y_train, y_test])
evaluators.fit(x, y)
# print scores of each estiamtor
means = evaluators.cv_results_['mean_test_score']
stds = evaluators.cv_results_['std_test_score']
times = evaluators.cv_results_['mean_fit_time']

for mean, std, time, params in zip(means, stds, times, evaluators.cv_results_['params']):
    print("%0.3f (+/-%0.03f), time: %0.03f for %r" % (mean, std * 2, time, params))
