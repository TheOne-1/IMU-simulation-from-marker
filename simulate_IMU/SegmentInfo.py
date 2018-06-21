from DatabaseInfo import DatabaseInfo
import numpy as np


class SegmentInfo:
    def __init__(self, segment_name):
        self.__segment_name = segment_name
        self.__center_marker_names = DatabaseInfo.get_center_marker_names(segment_name)

    def get_center_marker_names(self):
        return self.__center_marker_names

    def get_segment_name(self):
        return self.__segment_name

    def get_segment_R(self):
        R = {
            'trunk': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # R for trunk
            'pelvis': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # R for trunk
            'l_thigh': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # R for trunk
            'r_thigh': np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),  # R for trunk
            'l_shank': np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),  # R for trunk
            'r_shank': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # R for trunk
            'l_feet': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # R for trunk
            'r_feet': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # R for trunk
        }
        return R[self.__segment_name]

    def set_marker_cali_matrix(self, cali_data_df):
        marker_names = DatabaseInfo.get_segment_marker_names(self.__segment_name)
        marker_pos = cali_data_df[marker_names].as_matrix()
        marker_pos_mean = np.mean(marker_pos, axis=0)
        segment_marker_num = int(marker_names.__len__() / 3)
        marker_matrix = marker_pos_mean.reshape([segment_marker_num, 3])
        self.__marker_matrix = marker_matrix

    def get_marker_cali_matrix(self):
        return self.__marker_matrix


