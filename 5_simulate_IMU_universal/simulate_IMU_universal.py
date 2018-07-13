# this file is used to evaluate all the thigh marker position and find the best location

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from EvaluationClass import Evaluation
from const import *
import sklearn
from sklearn import ensemble, neighbors, tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from XYGenerator_uni import XYGenerator_uni
import numpy as np


X_NORM_ALL = True     # if true, normalize along all the subjects, if false, normalize each subject independently
Y_NORM = False        # if true, normalize along all the subjects, if false, do not do normalize

output_names = [
    'FP1.ForX', 'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
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
# moved_segments = MOVED_SEGMENT_NAMES
my_database_info = DatabaseInfo()
total_score_df = Evaluation.initialize_result_df(total_result_columns)

# model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=4)
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=4000000)
model = ensemble.GradientBoostingRegressor(
    learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)

x_scalar, y_scalar = preprocessing.StandardScaler(), preprocessing.StandardScaler()
for segment_moved in SEGMENT_NAMES[0:6]:
    print('Segment: ' + segment_moved)  # to show how much data have been processed
    # get all subjects' data
    x, y = {}, {}
    for i_sub in range(SUB_NUM):
        subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
        x_sub, y_sub = {}, {}
        for speed in SPEEDS:
            my_xy_generator = XYGenerator_uni(subject_data, segment_moved, speed, output_names, input_names)
            x_trial, y_trial = my_xy_generator.get_xy()
            x_sub[speed] = x_trial.as_matrix()
            y_sub[speed] = y_trial.as_matrix()

        if not X_NORM_ALL:
            x_scalar.fit(x_sub)
            x_sub = x_scalar.transform(x_sub)

        x[i_sub] = x_sub
        y[i_sub] = y_sub

    for i_sub_test in range(SUB_NUM):
        print('\tSubject: ' + str(i_sub_test))
        # get training data
        x_train, y_train = np.zeros([0, input_names.__len__()]), np.zeros([0, input_names.__len__()])
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
            y_train = y_scalar


            # get testing data
        # for speed in SPEEDS:
        #     subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub_test)
        #     my_xy_generator = XYGenerator_uni(subject_data, segment_moved, speed, output_names, input_names)
        #     x_test = x[i_sub_test][speed]
        #     y_test = y[i_sub_test][speed]
        #     score_list = my_xy_generator.get_score_list(x_train, x_test, y_train, y_test, model)

# # get testing data
#
#             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
#             # my_xy_generator.check_acc_difference(x)
#             score_list = my_xy_generator.get_score_list(x_train, x_test, y_train, y_test, model)
#             # Evaluation.show_result(score_list, my_xy_generator, output_names)
#             total_score_df = total_score_df.append(
#                 Evaluation.store_current_result(score_list, my_xy_generator, total_result_columns))
# Evaluation.save_total_result(total_score_df, input_names, output_names, model)
