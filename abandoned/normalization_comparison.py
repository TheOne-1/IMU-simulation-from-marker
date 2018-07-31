# this file is used to evaluate all the thigh marker position and find the best location

import numpy as np
from sklearn import ensemble
from sklearn import preprocessing

from DatabaseInfo import DatabaseInfo
from EvaluationUniClass import EvaluationUni
from SubjectData import SubjectData
from XYGeneratorUni import XYGeneratorUni
from const import *

output_names = [
    'FP1.ForX', 'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
]
total_result_columns = ['subject_id', 'X_NORM_ALL', 'Y_NORM']
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
# moved_segments = MOVED_SEGMENT_NAMES
my_database_info = DatabaseInfo()
total_score_df = EvaluationUni.initialize_result_df(total_result_columns)

# model = ensemble.RandomForestRegressor(n_jobs=4)
# model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4)
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=4000000)
model = ensemble.GradientBoostingRegressor(
    learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)
x_scalar, y_scalar = preprocessing.StandardScaler(), preprocessing.StandardScaler()

# # for test
# SUB_NUM = 2

X_NORM_ALL = True     # if true, normalize along all the subjects, if false, normalize each subject independently
Y_NORM = False        # if true, normalize along all the subjects, if false, do not do normalize

for X_NORM_ALL in [False]:
    for Y_NORM in [True]:
        print('X_NORM_ALL: ' + str(X_NORM_ALL) + '\tY_NORM: ' + str(Y_NORM))
        x, y = {}, {}
        for i_sub in range(SUB_NUM):
            subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
            x_sub, y_sub = {}, {}
            for speed in SPEEDS:
                my_xy_generator = XYGeneratorUni(subject_data, SEGMENT_NAMES[0], speed, output_names, input_names)
                x_trial, y_trial = my_xy_generator.get_xy()
                x_sub[speed] = x_trial.as_matrix()
                y_sub[speed] = y_trial.as_matrix()
            if not X_NORM_ALL:
                x_scalar.fit(np.row_stack([x_sub[SPEEDS[0]], x_sub[SPEEDS[1]], x_sub[SPEEDS[2]]]))
                for speed in SPEEDS:
                    x_sub[speed] = x_scalar.transform(x_sub[speed])
            x[i_sub] = x_sub
            y[i_sub] = y_sub

        for i_sub_test in range(1):
            print('subject: ' + str(i_sub_test))
            # get training data
            x_train, y_train = np.zeros([0, input_names.__len__()]), np.zeros([0, output_names.__len__()])
            for i_sub in range(SUB_NUM):
                if i_sub != i_sub_test:     # test data will not be added into train data
                    for speed in SPEEDS:
                        x_train = np.row_stack([x_train, x[i_sub][speed]])
                        y_train = np.row_stack([y_train, y[i_sub][speed]])

            x_test = np.row_stack([x[i_sub_test][SPEEDS[0]], x[i_sub_test][SPEEDS[1]], x[i_sub_test][SPEEDS[2]]])
            y_test = np.row_stack([y[i_sub_test][SPEEDS[0]], y[i_sub_test][SPEEDS[1]], y[i_sub_test][SPEEDS[2]]])

            if X_NORM_ALL:
                x_scalar.fit(x_train)
                x_train = x_scalar.transform(x_train)
                x_test = x_scalar.transform(x_test)
            else:
                x_test = x_scalar.transform(x_test)

            if Y_NORM:
                y_scalar.fit(y_train)
                y_train = y_scalar.transform(y_train)

            my_evaluator = EvaluationUni(x_train, x_test, y_train, y_test, output_names, model)
            my_evaluator.train_sklearn()
            y_pred = my_evaluator.evaluate_sklearn()

            if Y_NORM:
                y_pred = y_scalar.inverse_transform(y_pred)
            scores = EvaluationUni.get_scores(y_test, y_pred)
            df_item = EvaluationUni.scores_df_item(scores, total_result_columns, i_sub_test, X_NORM_ALL, Y_NORM)
            total_score_df = total_score_df.append(df_item)

EvaluationUni.save_result(total_score_df, input_names, output_names, model, X_NORM_ALL, Y_NORM)
