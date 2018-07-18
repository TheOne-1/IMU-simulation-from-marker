# this file is used to evaluate all the thigh marker position and find the best location
# it is combines methods such as Filter, Wrapper and embedded feature importance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from XYGeneratorUni import XYGeneratorUni
from const import *
from FeatureModelSelector import FeatureModelSelector

from sklearn.utils import shuffle
from sklearn import ensemble, neighbors, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVR

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY',
    'FP2.ForY',
    'FP1.ForZ',
    'FP2.ForZ',
    'FP1.CopX',
    'FP1.CopY',
    'FP2.CopX',
    'FP2.CopY'
]

input_names = [
    # 'mass', 'height',
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
# moved_segments = MOVED_SEGMENT_NAMES

# model = ensemble.RandomForestRegressor(n_jobs=4)
# model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4)
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=4000000)
model = ensemble.GradientBoostingRegressor(
    learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)
x_scalar, y_scalar = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

path = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data_all_sub\\all_sub.csv'
all_data = pd.read_csv(path, index_col=False)

# normalize x
x, y = all_data[input_names].as_matrix(), all_data[output_names].as_matrix()
x = x_scalar.fit_transform(x)
y = y_scalar.fit_transform(y)

# shuffle data
x, y = shuffle(x, y)
x = x[:20000, :]
y = y[:20000, :]

column_names = input_names.copy()
column_names.extend(['method', 'score_func', 'output_name'])
selection_result_df = pd.DataFrame()

for i_output in range(len(output_names)):
    print('output number: ' + str(i_output))
    selector = SelectKBest(f_regression, k=1)
    selector.fit(x, y[:, i_output])
    scores = np.array(selector.scores_)
    scores = FeatureModelSelector.normalize_array(scores, axis=0)
    df_item = pd.DataFrame([scores])
    df_item['method'] = 'SelectKBest'
    df_item['score_func'] = 'f_regression'
    df_item['output_name'] = output_names[i_output]
    selection_result_df = selection_result_df.append(df_item)

    # selector = SelectKBest(mutual_info_regression, k=24)
    # selector.fit(x, y[:, i_output])
    # scores = selector.scores_
    # df_item = pd.DataFrame([scores])
    # df_item['method'] = 'SelectKBest'
    # df_item['score_func'] = 'mutual_info_regression'
    # df_item['output_name'] = output_names[i_output]
    # selection_result_df = selection_result_df.append(df_item)
    # print('mutual_info_regression')

    selector = RFE(ensemble.GradientBoostingRegressor(
        learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500), n_features_to_select=1)
    selector.fit(x, y[:, i_output])
    scores = np.array(100 / selector.ranking_)
    scores = FeatureModelSelector.normalize_array(scores, axis=0)
    df_item = pd.DataFrame([scores])
    df_item['method'] = 'RFE'
    df_item['score_func'] = 'GradientBoostingRegressor'
    df_item['output_name'] = output_names[i_output]
    selection_result_df = selection_result_df.append(df_item)

    # selector = RFE(estimator=SVR(kernel='linear', C=200, epsilon=0.02, gamma=0.1, max_iter=4000),
    #                n_features_to_select=1)
    # selector.fit(x, y[:, i_output])
    # scores = 100 / selector.ranking_
    # df_item = pd.DataFrame([scores])
    # df_item['method'] = 'RFE'
    # df_item['score_func'] = 'SVR'
    # df_item['output_name'] = output_names[i_output]
    # selection_result_df = selection_result_df.append(df_item)

    model = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=6)
    model.fit(x, y[:, i_output])
    scores = np.array(model.feature_importances_)
    scores = FeatureModelSelector.normalize_array(scores, axis=0)
    df_item = pd.DataFrame([scores])
    df_item['method'] = 'SelectFromModel'
    df_item['score_func'] = 'RandomForest'
    df_item['output_name'] = output_names[i_output]
    selection_result_df = selection_result_df.append(df_item)


selection_result_df.columns = column_names
selection_result_df.to_csv(RESULT_PATH + 'importance_matrix\\feature_selection.csv', index=False)










# # SelectKBest, f_regression
# selector = SelectKBest(f_regression, k=24)
# FeatureModelSelector.show_selection_outputs(selector, x, y, input_names, output_names)
# #
# # # SelectKBest, mutual_info_regression
# selector = SelectKBest(mutual_info_regression, k=24)
# FeatureModelSelector.show_selection_outputs(selector, x, y, input_names, output_names)
#
# # # RFE, RandomForestRegressor
# selector = RFE(estimator=ensemble.RandomForestRegressor(), n_features_to_select=1)
# FeatureModelSelector.show_selection_outputs(selector, x, y, input_names, output_names)
#
# # RFE, SVR
# selector = RFE(estimator=SVR(kernel='linear', C=200, epsilon=0.02, gamma=0.1, max_iter=4000), n_features_to_select=1)
# FeatureModelSelector.show_selection_outputs(selector, x, y, input_names, output_names)








# selector = RFECV(estimator=ensemble.RandomForestRegressor())
# feature_selected = np.zeros([len(output_names), len(input_names)])
# plt.figure()
# for i_output in range(len(output_names)):
#     # selector.fit(x, y[:, i_output])
#     selector.fit(x[:6000, :], y[:6000, i_output])
#     trial_scores = 100 / selector.ranking_
#     feature_selected[i_output, :] = trial_scores
#     plt.plot(feature_selected[i_output, :], label=output_names[i_output])
#     plt.plot(range(1, len(input_names)+1), selector.grid_scores_, label=output_names[i_output])
#     plt.xticks(range(1, len(input_names)+1))
