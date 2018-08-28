# this class is used to get the subject data

import pandas as pd
from const import *
from DatabaseInfo import DatabaseInfo


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

    def get_subject_id(self):
        return self.__subject_id

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
    @staticmethod
    def get_cylinder_diameter(df, segment):
        if segment in ['l_thigh', 'r_thigh']:
            return THIGH_COEFF * df['knee_width'].as_matrix()[0]
        if segment in ['l_shank', 'r_shank']:
            knee_width = df['knee_width'].as_matrix()[0]
            ankle_width = df['ankle_width'].as_matrix()[0]
            return (knee_width + ankle_width) / 2

























