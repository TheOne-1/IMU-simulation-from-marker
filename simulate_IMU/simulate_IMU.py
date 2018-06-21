# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

from DatabaseInfo import DatabaseInfo
from VirtualProcessor import Processor
from SubjectDataGetter import SubjectDataGetter
from EvaluationClass import Evaluation
from const import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import ensemble
import numpy as np


output_names = [
    'FP1.ForX', 'FP2.ForX',
    # 'FP1.ForY', 'FP2.ForY',
    # 'FP1.ForZ', 'FP2.ForZ',
    # 'FP1.CopX', 'FP1.CopY',
    # 'FP2.CopX', 'FP2.CopY'
]

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
my_processor = Processor(output_names, input_names)

for segment_moved in SEGMENT_NAMES:
    for i_sub in range(SUB_NUM):
        subject_data = SubjectDataGetter(PROCESSED_DATA_PATH, i_sub)
        for i_speed in range(SPEED_NUM):
            # get walking 1 data
            x, y = my_processor.get_xy(subject_data, SPEEDS[i_speed])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
            point_list, x_range, y_range = my_processor.get_simulate_position(subject_data, SPEEDS[i_speed], segment_moved)
            # fake_list = my_processor.get_fake_position(subject_data, SPEEDS[i_speed], segment_moved)    # !!!
            # point_list = fake_list    # !!!
            score_list = []
            for simulated_marker in point_list:
                x_test = my_processor.modify_acc_test(subject_data, SPEEDS[i_speed], simulated_marker, x_test)
                my_evaluator = Evaluation(x_train, x_test, y_train, y_test, output_names)
                model_random_forest = ensemble.RandomForestRegressor(random_state=10)
                my_evaluator.train_sklearn(model_random_forest)
                score_list.append(my_evaluator.evaluate_sklearn(show_plot=False))

            my_processor.show_result(score_list)





            # my_processor.set_segment(segment_moved)
            # cali_data_df, walking_data_1_df, walking_data_2_df = subject_data.get_all_data(SPEEDS[i_speed])
            # my_processor.set_marker_cali_matrix(cali_data_df)
            #
            # # the point list that will generate virtual marker
            # point_list = my_processor.get_point_list(cali_data_df)
            #
            # # point_list = [my_processor.get_center_point_mean(cali_data_df)]   #!!! for virtual marker checking
            # for simulated_marker in point_list:
            #     virtual_marker, R_IMU_transform = my_processor. \
            #         get_virtual_marker(simulated_marker, walking_data_1_df)
            #     my_processor.check_virtual_marker(virtual_marker, walking_data_1_df)
            #     acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)
