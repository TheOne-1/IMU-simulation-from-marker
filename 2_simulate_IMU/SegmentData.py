from DatabaseInfo import DatabaseInfo
import numpy as np
from const import *


class SegmentData:
    def __init__(self, subject_data, segment_name):
        self.__segment_name = segment_name
        self.__center_marker_names = DatabaseInfo.get_center_marker_names(segment_name)
        self.__segment_cali_data = {}
        self.__segment_walking_1_data = {}
        self.__segment_walking_2_data = {}
        self.__marker_cali_matrix = {}
        self.__center_point_mean = {}
        for speed in SPEEDS:
            self.__segment_cali_data[speed] = subject_data.get_cali_segment_data(speed, segment_name)
            self.__segment_walking_1_data[speed] = subject_data.get_walking_1_segment_data(speed, segment_name)
            self.__segment_walking_2_data[speed] = subject_data.get_walking_2_segment_data(speed, segment_name)

            marker_pos = self.__segment_cali_data[speed].as_matrix()
            marker_pos_mean = np.mean(marker_pos, axis=0)
            segment_marker_num = int(marker_pos.shape[1] / 3)
            marker_matrix = marker_pos_mean.reshape([segment_marker_num, 3])
            self.__marker_cali_matrix[speed] = marker_matrix

            center_point = self.__segment_cali_data[speed][self.__center_marker_names].as_matrix()
            center_point_mean = np.mean(center_point, axis=0)
            center_marker_num = int(center_point_mean.shape[0] / 3)
            if center_marker_num != 1:
                center_point_mean = (center_point_mean[0:3] + center_point_mean[3:6]) / 2
            self.__center_point_mean[speed] = center_point_mean

    def get_center_point_mean(self, speed):
        return self.__center_point_mean[speed]

    def get_segment_name(self):
        return self.__segment_name

    def get_segment_R(self):
        R = {
            'trunk': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # R for trunk
            'pelvis': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # R for pelvis
            'l_thigh': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # R for l_thigh
            'r_thigh': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),  # R for r_thigh
            'l_shank': np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),  # R for l_shank
            'r_shank': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # R for r_shank
            'l_feet': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # R for l_feet
            'r_feet': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # R for r_feet
        }
        return R[self.__segment_name]

    def get_marker_cali_matrix(self, speed):
        return self.__marker_cali_matrix[speed]

    def get_segment_walking_1_data(self, speed):
        return self.__segment_walking_1_data[speed]

    def get_segment_walking_2_data(self, speed):
        return self.__segment_walking_2_data[speed]


