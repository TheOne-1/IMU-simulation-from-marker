# this file is used to evaluate all the thigh marker position and find the best location

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from EvaluationUniClass import EvaluationUni
from const import *
import sklearn
from sklearn import ensemble, neighbors, tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from XYGeneratorUni import XYGeneratorUni
import numpy as np
from OffsetClass import *


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

trunk_range_x, trunk_range_z = XYGeneratorUni.get_move_range('trunk')
offset_axis_1 = OneAxisOffset('trunk', 'x', trunk_range_x)
offset_axis_2 = OneAxisOffset('pelvis', 'z', trunk_range_z)
multi_offset_axis = MultiAxisOffset()
multi_offset_axis.add_offset_axis(offset_axis_1)
multi_offset_axis.add_offset_axis(offset_axis_2)
offset_list = multi_offset_axis.get_combos()

for i_sub in range(SUB_NUM):
    print('\tSubject: ' + str(i_sub))
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    for speed in SPEEDS:
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        x, y = my_xy_generator.get_xy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
        test_index = x_test.index
        for offset_combo in offset_list:
            x_modified = my_xy_generator.modify_x(x, offset_combo)
            x_test = x_modified.loc[test_index]


















