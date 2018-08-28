# this file is used to evaluate all the thigh marker position and find the best location
import numpy as np
import pandas as pd
from gaitanalysis.motek import DFlowData

from DatabaseInfo import DatabaseInfo
from SubjectData import SubjectData
from XYGenerator import XYGeneratorUni
from const import *

data_path = 'D:\Tian\Research\Projects\ML Project\gait_database\GaitDatabase\data\\'

output_names = [
    'FP1.ForX', 'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
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
    'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
    'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
    'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
]
column_names = []
column_names.extend(input_names)
column_names.extend(output_names)
column_names.extend(['speed', 'subject_id', 'age', 'mass', 'height', 'knee_width', 'ankle_width'])

all_data_df = pd.DataFrame()
for i_sub in range(SUB_NUM):
    print(str(i_sub))
    file_names = DatabaseInfo.get_file_names(sub=i_sub, speed=0, path=data_path)
    subject_data_dflow = DFlowData(file_names[0], file_names[1], file_names[2])
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    x_sub, y_sub = {}, {}
    subject_data_df = pd.DataFrame()
    mass = subject_data_dflow.meta['subject']['mass']
    height = subject_data_dflow.meta['subject']['height']
    age = subject_data_dflow.meta['subject']['age']
    for speed in SPEEDS:
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        x_trial, y_trial = my_xy_generator.get_xy()
        if NORMALIZE_ACC:
            x_trial.loc[:, ALL_ACC_NAMES] /= height      # normalize acceleration by height
        if NORMALIZE_GRF:
            factor = mass * 9.8
            y_trial.loc[:, ALL_FORCE_NAMES] /= factor      # normalize GRF by mass
        speed_data = pd.DataFrame(np.column_stack([x_trial, y_trial]))
        speed_data['speed'] = speed
        subject_data_df = subject_data_df.append(speed_data)

    knee_width_left = subject_data_dflow.meta['subject']['knee-width-left'] / 1000
    knee_width_right = subject_data_dflow.meta['subject']['knee-width-right'] / 1000
    knee_width = (knee_width_left + knee_width_right) / 2
    ankle_width_left = subject_data_dflow.meta['subject']['ankle-width-left'] / 1000
    ankle_width_right = subject_data_dflow.meta['subject']['ankle-width-right'] / 1000
    ankle_width = (ankle_width_left + ankle_width_right) / 2
    subject_data_df['subject_id'] = i_sub
    subject_data_df['age'] = age
    subject_data_df['mass'] = mass
    subject_data_df['height'] = height
    subject_data_df['knee_width'] = knee_width
    subject_data_df['ankle_width'] = ankle_width

    all_data_df = all_data_df.append(subject_data_df)

all_data_df.columns = column_names
path = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data_all_sub\\all_sub.csv'
all_data_df.to_csv(path, index=False)










