# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from EvaluationClass import Evaluation
from const import *
from sklearn.model_selection import train_test_split
from TrainTestGenerator import TrainTestGenerator
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
    # 'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
    # 'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
    # 'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
    # 'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
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
segment_moved = SEGMENT_NAMES[1]        # segment_moved must be contained in the input_names

for i_sub in range(SUB_NUM):
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    for speed in SPEEDS:
        my_xy_generator = TrainTestGenerator(subject_data, segment_moved, speed, output_names)
        x_raw, y_raw = my_xy_generator.get_xy()
        x, y = x_raw[input_names], y_raw[output_names]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

        score_list = my_xy_generator.get_score_list(x_train, x_test, y_train, y_test, do_scaling=False)
        Evaluation.show_result(score_list, my_xy_generator)
        Evaluation.save_result(score_list, my_xy_generator)














