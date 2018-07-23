# this file is used to evaluate all the thigh marker position and find the best location

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from EvaluationUniClass import EvaluationUni
from const import *
from sklearn import ensemble, neighbors, tree
from sklearn.svm import SVR
from sklearn import preprocessing
from XYGeneratorUni import XYGeneratorUni
import numpy as np
from OffsetClass import *
import pandas as pd

output_names = [
    'FP1.ForX',
    'FP2.ForX',
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
total_score_df = EvaluationUni.initialize_result_df(total_result_columns)

# model = ensemble.RandomForestRegressor(n_jobs=6)
model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=6)
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=400)
# model = ensemble.GradientBoostingRegressor(
#     learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)
x_scalar, y_scalar = preprocessing.StandardScaler(), preprocessing.StandardScaler()

all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
total_result_df = pd.DataFrame()

for i_sub_test in range(SUB_NUM):
    print('subject: ' + str(i_sub_test))
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub_test]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = EvaluationUni(output_names, model)
    my_evaluator.set_train(x_train, y_train)

    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub_test)

    # the range have to be defined after subject data to get the diameter

    shank_diameter = subject_data.get_cylinder_diameter('l_shank')
    thigh_diameter = subject_data.get_cylinder_diameter('l_thigh')
    multi_offset_axis = MultiAxisOffset()

    # offset_axis_1 = OneAxisOffset('trunk', 'x', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_1)
    # offset_axis_2 = OneAxisOffset('trunk', 'z', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_2)
    # offset_axis_3 = OneAxisOffset('pelvis', 'x', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_3)
    # offset_axis_4 = OneAxisOffset('pelvis', 'z', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_4)
    # offset_axis_5 = OneAxisOffset('l_thigh', 'z', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_5)
    # offset_axis_6 = OneAxisOffset('l_thigh', 'theta', range(-30, 31, 30), shank_diameter)
    # multi_offset_axis.add_offset_axis(offset_axis_6)
    # offset_axis_7 = OneAxisOffset('r_thigh', 'z', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_7)
    offset_axis_8 = OneAxisOffset('r_thigh', 'theta', range(-30, 31, 30), shank_diameter)
    multi_offset_axis.add_offset_axis(offset_axis_8)
    offset_axis_9 = OneAxisOffset('l_shank', 'z', range(-100, 101, 100))
    multi_offset_axis.add_offset_axis(offset_axis_9)
    offset_axis_10 = OneAxisOffset('l_shank', 'theta', range(-30, 31, 30), shank_diameter)
    multi_offset_axis.add_offset_axis(offset_axis_10)
    # offset_axis_11 = OneAxisOffset('r_shank', 'z', range(-100, 101, 100))
    # multi_offset_axis.add_offset_axis(offset_axis_11)
    # offset_axis_12 = OneAxisOffset('r_shank', 'theta', range(-30, 31, 30), shank_diameter)
    # multi_offset_axis.add_offset_axis(offset_axis_12)

    offset_list = multi_offset_axis.get_combos()
    combo_len = len(offset_list)  # the length of the combos
    all_offsets_df = multi_offset_axis.get_offset_df()

    my_evaluator.train_sklearn()        # very time consuming

    for speed in SPEEDS:
        test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub_test]
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names]
        my_evaluator.set_y_test(y_test)
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        scores = np.zeros([combo_len, len(output_names)])
        for i_combo in range(len(offset_list)):
            x_test_modified = my_xy_generator.modify_x(x_test, offset_list[i_combo])
            my_evaluator.set_x_test(x_test_modified)
            y_pred = my_evaluator.evaluate_sklearn()
            scores[i_combo, :] = my_evaluator.get_scores()
            # for offset in offset_list[i_combo]:
            #     offset.to_string()
            # print('scores: ' + str(scores[i_combo]) + '\n')
        scores_df = pd.DataFrame(scores, columns=output_names)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub_test)
        total_result_df = pd.concat([total_result_df, speed_df], axis=0)

EvaluationUni.save_uni_result(total_result_df, model, input_names, output_names)




# for i_sub in range(SUB_NUM):
#     print('\tSubject: ' + str(i_sub))
#     subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
#     for speed in SPEEDS:
#         x, y = my_xy_generator.get_xy()
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
#         test_index = x_test.index
#         for offset_combo in offset_list:
#             x_modified = my_xy_generator.modify_x(x, offset_combo)
#             x_test = x_modified.loc[test_index]


















