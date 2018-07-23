# this class is used to get the subject data

import pandas as pd
from const import *
from DatabaseInfo import DatabaseInfo
import numpy as np
from gaitanalysis.motek import DFlowData

class SubjectData:

    def __init__(self, processed_data_path, subject_id):
        file_names = DatabaseInfo.get_file_names(sub=subject_id, speed=0, path=RAW_DATA_PATH)
        subject_data_dflow = DFlowData(file_names[0], file_names[1], file_names[2])
        self.__mass = subject_data_dflow.meta['subject']['mass']
        self.__age = subject_data_dflow.meta['subject']['age']
        self.__height = subject_data_dflow.meta['subject']['height']
        self.__gender = subject_data_dflow.meta['subject']['gender']
        knee_width_left = subject_data_dflow.meta['subject']['knee-width-left'] / 1000
        knee_width_right = subject_data_dflow.meta['subject']['knee-width-right'] / 1000
        self.__knee_width = (knee_width_left + knee_width_right) / 2
        ankle_width_left = subject_data_dflow.meta['subject']['ankle-width-left'] / 1000
        ankle_width_right = subject_data_dflow.meta['subject']['ankle-width-right'] / 1000
        self.__ankle_width = (ankle_width_left + ankle_width_right) / 2

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
    def get_cylinder_diameter(self, segment):
        if segment in ['l_thigh', 'r_thigh']:
            return THIGH_COEFF * self.__knee_width
        if segment in ['l_shank', 'r_shank']:
            return (self.__knee_width + self.__ankle_width) / 2
























