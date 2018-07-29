# this file is used to generate virtual marker and IMU
import math
import numpy as np
from Initializer import Initializer
import matplotlib.pyplot as plt
import scipy.interpolate as interpo
from const import *
from DatabaseInfo import DatabaseInfo

# 这个类原则上只能存放静态方法
class Processor:

    @staticmethod
    def rigid_transform_3D(A, B):
        return Initializer.rigid_transform_3D(A, B)

    # get virtual marker and R_IMU_transform
    @staticmethod
    def get_virtual_marker(simulated_marker, walking_data, marker_cali_matrix, R_standing_to_ground):
        segment_marker_num = marker_cali_matrix.shape[0]
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
    def check_virtual_marker(virtual_marker, walking_df, segment_name):
        center_marker_name = DatabaseInfo.get_center_marker_names(segment_name)
        center_marker_num = int(center_marker_name.__len__() / 3)
        # check the virtual marker data
        real_marker_pos = walking_df[center_marker_name].as_matrix()
        if center_marker_num != 1:
            real_marker_pos = (real_marker_pos[:, 0:3] + real_marker_pos[:, 3:6])/2
        error = virtual_marker - real_marker_pos
        error_combined = np.linalg.norm(error, axis=1)
        plt.figure()
        plt.plot(error_combined)
        plt.title('difference between real marker and virtual marker')
        plt.xlabel('frame')
        plt.ylabel('mm')
        plt.show()

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
        #
        # plt.figure()
        # plt.subplot(221)
        # plt.plot(acc_to_IMU[:, 0])
        # plt.subplot(222)
        # plt.plot(acc_to_IMU[:, 1])
        # plt.subplot(223)
        # plt.plot(acc_to_IMU[:, 2])
        # acc_to_IMU_norm = np.linalg.norm(acc_to_IMU, axis=1)
        # plt.subplot(224)
        # plt.plot(acc_to_IMU_norm)
        # plt.show()

        return acc_to_IMU

    @staticmethod
    def get_gyr(walking_data_df, R_IMU_transform):
        walking_data = walking_data_df.as_matrix()
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        gyr_middle = np.zeros([data_len, 3])
        for i_frame in range(data_len - 1):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame + 1, :].reshape([marker_number, 3])
            [R_one_sample, t] = Processor.rigid_transform_3D(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)
            a, b = np.linalg.eig(R_one_sample)
            for i_eig in range(a.__len__()):
                if abs(a[i_eig].imag) < 1e-12:
                    vector = b[:, i_eig].real
                    break
                if i_eig == a.__len__():
                    raise RuntimeError('no eig')

            if (R_one_sample[2, 1] - R_one_sample[1, 2]) * vector[0] < 0:  # check the direction of the rotation axis
                vector = -vector
            vector = np.dot(R_IMU_transform[:, :, i_frame].T, vector)
            gyr_middle[i_frame, :] = theta * vector * MOCAP_SAMPLE_RATE

        step_middle = np.arange(0.5 / MOCAP_SAMPLE_RATE, data_len / MOCAP_SAMPLE_RATE + 0.5 / MOCAP_SAMPLE_RATE,
                                1 / MOCAP_SAMPLE_RATE)
        step_gyr = np.arange(0, data_len / MOCAP_SAMPLE_RATE, 1 / MOCAP_SAMPLE_RATE)
        # in splprep, s the amount of smoothness. 6700 might be appropriate
        tck, step = interpo.splprep(gyr_middle.T, u=step_middle, s=0)
        gyr = interpo.splev(step_gyr, tck, der=0)
        gyr = np.column_stack([gyr[0], gyr[1], gyr[2]])


        # plt.figure()
        # plt.subplot(221)
        # plt.plot(gyr[:, 0])
        # plt.subplot(222)
        # plt.plot(gyr[:, 1])
        # plt.subplot(223)
        # plt.plot(gyr[:, 2])
        # gyr_norm = np.linalg.norm(gyr, axis=1)
        # plt.subplot(224)
        # plt.plot(gyr_norm)
        # plt.show()

        return gyr

    @staticmethod
    def get_cylinder_surface_rotation(theta):
        # theta should be in radians
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])








