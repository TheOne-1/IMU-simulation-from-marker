# this file is used to generate virtual marker and IMU

import numpy as np
from Initializer import Initializer
import matplotlib.pyplot as plt
import scipy.interpolate as interpo
from const import *


# 这个类原则上只能存放静态方法
class Processor:

    @staticmethod
    def rigid_transform_3D(A, B):
        return Initializer.rigid_transform_3D(A, B)

    # get virtual marker and R_IMU_transform
    @staticmethod
    def get_virtual_marker(simulated_marker, walking_data, marker_cali_matrix, R_standing_to_ground):
        # segment_name = self.__segment_info.get_segment_name()
        # marker_cali_matrix = self.__segment_info.get_marker_cali_matrix()
        # R_standing_to_ground = self.__segment_info.get_segment_R()
        segment_marker_num = marker_cali_matrix.shape[0]
        #
        # segment_marker = DatabaseInfo.get_segment_marker_names(segment_name)
        # walking_data = walking_data_df[segment_marker].as_matrix()
        walking_data = walking_data.as_matrix()
        data_len = walking_data.shape[0]
        virtual_marker = np.zeros([data_len, 3])
        R_IMU_transform = np.zeros([3, 3, data_len])
        for i_frame in range(data_len):
            current_marker_matrix = walking_data[i_frame, :].reshape([segment_marker_num, 3])
            [R_between_frames, t] = Initializer.rigid_transform_3D(marker_cali_matrix, current_marker_matrix)
            virtual_marker[i_frame, :] = (np.dot(R_between_frames, simulated_marker) + t)
            R_IMU_transform[:, :, i_frame] = np.matmul(R_standing_to_ground, R_between_frames.T)
        return virtual_marker, R_IMU_transform

    # 和setsegment联系过于紧密
    @staticmethod
    def check_virtual_marker(virtual_marker, real_marker_pos):
        # segment_name = self.__segment_info.get_segment_name()
        # center_marker_name = DatabaseInfo.get_center_marker_names(segment_name)
        # center_marker_num = int(center_marker_name.__len__() / 3)
        # # check the virtual marker data
        # real_marker_pos = walking_data_df[center_marker_name].as_matrix()
        # if center_marker_num != 1:
        #     real_marker_pos = (real_marker_pos[:, 0:3] + real_marker_pos[:, 3:6])/2
        error = virtual_marker - real_marker_pos
        error_combined = np.linalg.norm(error, axis=1)
        plt.figure()
        plt.plot(error_combined)
        plt.title('difference between real marker and virtual marker')
        plt.xlabel('frame')
        plt.ylabel('mm')
        plt.show()
        x = 1

    @staticmethod
    def get_acc(virtual_marker, R_IMU_transform):
        data_len = virtual_marker.shape[0]
        # get acceleration with gravity in the ground frame
        step_marker = np.arange(0, data_len / MOCAP_SAMPLE_RATE, 1 / MOCAP_SAMPLE_RATE)
        # in splprep, s the amount of smoothness. 6700 might be appropriate
        tck, step_marker = interpo.splprep(virtual_marker.T, u=step_marker, s=0)
        acc_no_g = interpo.splev(step_marker, tck, der=2)  # der=2 means take the second derivation
        acc_no_g = np.column_stack([acc_no_g[0], acc_no_g[1], acc_no_g[2]])
        acc_to_ground = acc_no_g
        acc_to_ground[:, 2] = acc_to_ground[:, 2] + 9.81  # 9.8 could be changed

        # transfer the acceleration to the IMU frame
        acc_to_IMU = np.zeros([data_len, 3])
        for i_frame in range(data_len):
            acc_to_IMU[i_frame, :] = np.dot(R_IMU_transform[:, :, i_frame], acc_to_ground[i_frame, :].T)
        return acc_to_IMU
