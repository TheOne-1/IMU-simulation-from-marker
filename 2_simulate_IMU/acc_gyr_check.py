# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from EvaluationClass import Evaluation
from const import *
from XYGenerator import XYGenerator
import matplotlib.pyplot as plt

def plot_acc(acc):
    # compare the real and simulated acc
    plt.figure()
    plt.subplot(221)
    plt.plot(acc[:, 0])
    plt.title('x  ')
    plt.subplot(222)
    plt.plot(acc[:, 1])
    plt.title('y  ')
    plt.subplot(223)
    plt.plot(acc[:, 2])
    plt.title('z  ')

output_names = [
    'FP1.ForX',
    # 'FP2.ForX',
    # 'FP1.ForY', 'FP2.ForY',
    # 'FP1.ForZ', 'FP2.ForZ',
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

segment_moved = SEGMENT_NAMES[0]
i_sub = 1
speed = SPEEDS[0]

subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
my_xy_generator = XYGenerator(subject_data, segment_moved, speed, output_names, input_names)
x_raw, y_raw = my_xy_generator.get_xy()
x, y = x_raw[input_names], y_raw[output_names]
x = x.as_matrix()

sensor_num = int(input_names.__len__() / 3)
for i_sensor in range(sensor_num):
    plot_acc(x[:, 3*i_sensor:3*(i_sensor+1)])

plt.show()















