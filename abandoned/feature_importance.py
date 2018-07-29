# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

import numpy as np
import sklearn
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.utils import shuffle

from DatabaseInfo import DatabaseInfo
from EvaluationClass import Evaluation
from FeatureModelSelector import FeatureModelSelector
from SubjectData import SubjectData
from XYGenerator import XYGenerator
from const import *

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX',
    'FP2.CopY'
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
    # 'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
    # 'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    # 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    # 'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    # 'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    # 'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    # 'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
    # 'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
]
my_database_info = DatabaseInfo()
total_score_df = Evaluation.initialize_result_df(total_result_columns)

base_model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4)
# base_model = GridSearchCV(SVR(tol=0.01), param_grid={'kernel': ['rbf', 'sigmoid', 'poly']})

segment_moved = SEGMENT_NAMES[0]
impt_trials = np.zeros([output_names.__len__(), input_names.__len__(), SUB_NUM * SPEED_NUM])
i_matrice = 0
for i_sub in range(SUB_NUM):
    print('subject: ' + str(i_sub))
    for i_speed in range(SPEED_NUM):
        subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
        my_xy_generator = XYGenerator(subject_data, segment_moved, SPEEDS[i_speed], output_names, input_names)
        x_raw, y_raw = my_xy_generator.get_xy()
        x, y = x_raw[input_names], y_raw[output_names]
        # input and output scaling
        x_scaler = preprocessing.StandardScaler()
        x_scaler.fit(x)
        x = x_scaler.transform(x)
        y_scaler = preprocessing.StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)
        # shuffle the data
        x, y = shuffle(x, y)

        impt_trial = np.zeros([0, input_names.__len__()])
        for i_output in range(output_names.__len__()):
            model = sklearn.clone(base_model)
            model.fit(x, y[:, i_output])
            trial_feature_importance = 100 * np.array(model.feature_importances_)
            impt_trial = np.row_stack([impt_trial, trial_feature_importance])
        impt_trials[:, :, i_matrice] = impt_trial
        i_matrice += 1


FeatureModelSelector.store_impt_matrix_trial(impt_trials, input_names, output_names)

impt_mean = np.mean(impt_trials, axis=2)
impt_std = np.std(impt_trials, axis=2)
FeatureModelSelector.store_impt_matrix(impt_mean, input_names, output_names)
FeatureModelSelector.store_std_matrix(impt_mean, impt_std, input_names, output_names)





















