# this class is used to get the subject data

import pandas as pd
from const import *
from DatabaseInfo import DatabaseInfo
import numpy as np


class SubjectData:

    def __init__(self, processed_data_path, subject_id):
        self.__subject_id = subject_id
        self.__processed_data_path = processed_data_path
        self.__cali_data = {}       # a dict for 3 speeds' cali data
        self.__walking_1_data = {}
        self.__walking_2_data = {}
        for speed in SPEEDS:
            cali_file_path = self.__processed_data_path + 'subject_' + str(
                self.__subject_id) + '\\' + speed + '_cali.csv'
            self.__cali_data[speed] = pd.read_csv(cali_file_path)
            walking_file_path_1 = self.__processed_data_path + 'subject_' + \
                str(self.__subject_id) + '\\' + speed + '_walking_1.csv'
            self.__walking_1_data[speed] = pd.read_csv(walking_file_path_1)
            walking_file_path_2 = self.__processed_data_path + 'subject_' + \
                str(self.__subject_id) + '\\' + speed + '_walking_2.csv'
            self.__walking_2_data[speed] = pd.read_csv(walking_file_path_2)

    def get_all_data(self, speed):
        return self.__cali_data[speed], self.__walking_1_data[speed], self.__walking_2_data[speed]

    def get_cali_data(self, speed):
        return self.__cali_data[speed]

    def get_walking_1_data(self, speed):
        return self.__walking_1_data[speed]

    def get_walking_2_data(self, speed):
        return self.__walking_2_data[speed]

    def get_cali_segment_data(self, speed, segment):
        segment_names = DatabaseInfo.get_segment_marker_names(segment)
        return self.__cali_data[speed][segment_names]

    def get_walking_1_segment_data(self, speed, segment):
        segment_names = DatabaseInfo.get_segment_marker_names(segment)
        return self.__walking_1_data[speed][segment_names]

    def get_walking_2_segment_data(self, speed, segment):
        segment_names = DatabaseInfo.get_segment_marker_names(segment)
        return self.__walking_2_data[speed][segment_names]

    # get the diameter of the shank and thigh
    def get_cylinder_diameter(self, segment):
        markers = ['LGTRO.PosX', 'LGTRO.PosY', 'LGTRO.PosZ', 'RGTRO.PosX', 'RGTRO.PosY', 'RGTRO.PosZ']
        data = self.get_cali_data(speed=SPEEDS[0])[markers].as_matrix()
        great_trochanter_vector = data[:, 0:3] - data[:, 3:6]
        # data_mean = np.mean(data, axis=0)
        great_trochanter_dis = np.linalg.norm(great_trochanter_vector, axis=1)
        great_trochanter_dis_mean = np.mean(great_trochanter_dis)

        if segment in ['l_thigh', 'r_thigh']:
            return THIGH_COEFF * great_trochanter_dis_mean

        if segment in ['l_shank', 'r_shank']:
            return SHANK_COEFF * great_trochanter_dis_mean
























