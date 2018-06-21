# this file is used to generate virtual marker and IMU

import numpy as np
from DatabaseInfo import DatabaseInfo
from Initializer import Initializer
from SegmentInfo import SegmentInfo
import matplotlib.pyplot as plt
import scipy.interpolate as interpo
from const import *
import pandas as pd


class Processor:

    def __init__(self, output_names, input_names):
        self.__output_names = output_names
        self.__input_names = input_names

    def set_segment(self, segment_name):
        self.__segment_info = SegmentInfo(segment_name)

    @staticmethod
    def rigid_transform_3D(A, B):
        return Initializer.rigid_transform_3D(A, B)

    @staticmethod
    def get_surface_points(center_point, segment):
        if segment in ['trunk', 'pelvis']:
            return Processor.get_plane_surface_points(center_point, segment)
        # elif segment in ['l_thigh', 'r_thigh']:
            # return self.

    # plane surface for trunk and pelvis
    @staticmethod
    def get_plane_surface_points(center_point, segment):
        # range are in millimeter
        x_range, y_range = [], []
        if segment == 'trunk':
            trunk_range = 80
            x_range = range(-trunk_range, trunk_range+1, 20)
            y_range = range(-trunk_range, trunk_range+1, 20)
        elif segment == 'pelvis':
            pelvis_x_range = 50
            pelvis_y_range = 50
            x_range = range(-pelvis_x_range, pelvis_x_range+1, 10)
            y_range = range(-pelvis_y_range, pelvis_y_range+1, 2)

        point_list = []
        for x in x_range:
            for y in y_range:
                point = center_point + [x, y, 0]
                point_list.append(point)
        return point_list, x_range, y_range

    # @staticmethod
    # def get_cylinder_surface(center_point, segment):
    #     # range are in millimeter
    #     if segment in ['l_shank', 'r_shank']:
    #         trunk_range = 80
    #         x_range = range(-trunk_range, trunk_range+1, 20)
    #         y_range = range(-trunk_range, trunk_range+1, 20)
    #     elif segment == 'pelvis':
    #         pelvis_x_range = 50
    #         pelvis_y_range = 50
    #         x_range = range(-pelvis_x_range, pelvis_x_range+1, 10)
    #         y_range = range(-pelvis_y_range, pelvis_y_range+1, 2)
    #
    #     point_list = []
    #     for x in x_range:
    #         for y in y_range:
    #             point = center_point + [x, y, 0]
    #             point_list.append(point)
    #     return point_list, x_range, y_range

    # 和setsegment联系过于紧密
    def get_center_point_mean(self, cali_data_df):
        segment_name = self.__segment_info.get_segment_name()
        center_marker_name = DatabaseInfo.get_center_marker_names(segment_name)
        center_point = cali_data_df[center_marker_name].as_matrix()
        center_point_mean = np.mean(center_point, axis=0)
        center_marker_num = int(center_point_mean.shape[0] / 3)
        if center_marker_num != 1:
            center_point_mean = (center_point_mean[0:3] + center_point_mean[3:6]) / 2
        return center_point_mean

    # 和setsegment联系过于紧密
    def get_point_list(self, cali_data_df):
        segment_name = self.__segment_info.get_segment_name()
        center_point_mean = self.get_center_point_mean(cali_data_df)
        point_list, x_range, y_range = self.get_surface_points(center_point_mean, segment_name)
        return point_list, x_range, y_range

    # 和setsegment联系过于紧密
    def set_marker_cali_matrix(self, cali_data_df):
        self.__segment_info.set_marker_cali_matrix(cali_data_df)

    # get virtual marker and R_IMU_transform
    # 和setsegment联系过于紧密
    def get_virtual_marker(self, simulated_marker, walking_data_df):
        segment_name = self.__segment_info.get_segment_name()
        marker_cali_matrix = self.__segment_info.get_marker_cali_matrix()
        R_standing_to_ground = self.__segment_info.get_segment_R()
        segment_marker_num = marker_cali_matrix.shape[0]

        segment_marker = DatabaseInfo.get_segment_marker_names(segment_name)
        walking_data = walking_data_df[segment_marker].as_matrix()
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
    def check_virtual_marker(self, virtual_marker, walking_data_df):
        segment_name = self.__segment_info.get_segment_name()
        center_marker_name = DatabaseInfo.get_center_marker_names(segment_name)
        center_marker_num = int(center_marker_name.__len__() / 3)
        # check the virtual marker data
        real_marker_pos = walking_data_df[center_marker_name].as_matrix()
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

    # 后期 get_xy 应当返回六轴加速度、陀螺仪的 Dataframe，以方便选择输入量
    def get_xy(self, subject_data, speed):
        cali_data_df, walking_data_1_df, walking_data_2_df = subject_data.get_all_data(speed)
        # for now we only use walking data 1
        data_len = walking_data_1_df.shape[0]
        input_len = self.__input_names.__len__()

        # get x
        x = np.zeros([data_len, input_len])
        i_segment = 0
        for segment_name in SEGMENT_NAMES:
            self.set_segment(segment_name)
            self.set_marker_cali_matrix(cali_data_df)
            center_marker = self.get_center_point_mean(cali_data_df)
            virtual_marker, R_IMU_transform = self.get_virtual_marker(center_marker, walking_data_1_df)
            # self.check_virtual_marker(virtual_marker, walking_data_1_df)    # for check
            acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)
            x[:, 3*i_segment:3*(i_segment+1)] = acc_IMU
            i_segment += 1

        x = pd.DataFrame(x)
        x.columns = self.__input_names

        # get y
        y = walking_data_1_df[self.__output_names]

        return x, y

    def get_simulate_position(self, subject_data, speed, segment_moved):
        self.set_segment(segment_moved)
        cali_data_df = subject_data.get_cali_data(speed)
        self.set_marker_cali_matrix(cali_data_df)
        # the point list that will generate virtual marker
        point_list, x_range, y_range = self.get_point_list(cali_data_df)
        return point_list, x_range, y_range

    # !!! for test
    def get_fake_position(self, subject_data, speed, segment_moved):
        self.set_segment(segment_moved)
        cali_data_df = subject_data.get_cali_data(speed)
        self.set_marker_cali_matrix(cali_data_df)
        center_point_mean = self.get_center_point_mean(cali_data_df)
        temp = np.array([0.1, 0.1, 0.1])
        point_list = [center_point_mean - temp, center_point_mean, center_point_mean + temp]
        return point_list

    def modify_acc_test(self, subject_data, speed, simulated_marker, x_test):
        walking_data_1_df = subject_data.get_walking_1_data(speed)
        test_index = x_test.index

        virtual_marker, R_IMU_transform = self.get_virtual_marker(simulated_marker, walking_data_1_df)
        acc_IMU = self.get_acc(virtual_marker, R_IMU_transform)

        changed_columns = []
        for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
            column = self.__segment_info.get_segment_name() + acc_name
            changed_columns.append(column)

        acc_IMU_df = pd.DataFrame(acc_IMU)
        temp = acc_IMU_df.loc[test_index]
        x_test[changed_columns] = temp
        return x_test

    def show_result(self, score_list, show_plot=True):
        x_range, y_range = Processor.__get_move_range(self.__segment_info.get_segment_name())
        i_pos = 0
        result = np.zeros([score_list.__len__(), 3])
        for x in x_range:
            for y in y_range:
                result[i_pos, :] = [x, y, score_list[i_pos]]
                i_pos += 1
        result_df = pd.DataFrame(result)
        result_df.columns = ['x', 'y', 'score']
        result_df.to_csv('D:\Tian\Research\Projects\ML Project\simulatedIMU\python\\0517GaitDatabase\data_' +
                         self.__segment_info.get_segment_name() + '.csv')

        if show_plot:
            # change result as an image
            result_im = np.zeros([x_range.__len__(), y_range.__len__()])
            i_y, i_result = 0, 0
            for x in x_range:
                i_x = 0
                for y in y_range:
                    result_im[i_x, i_y] = score_list[i_result]
                    i_result += 1
                    i_x += 1
                i_y += 1

            fig, ax = plt.subplots()
            plt.imshow(result_im)
            x_label = list(x_range)
            x_label.insert(0, 0)        # 不知道为什么set_xticklabels会忽略第一个，所以在头部添加一个元素
            ax.set_xticklabels(x_label)
            y_label = list(y_range)
            y_label.insert(0, 0)        # 不知道为什么set_xticklabels会忽略第一个，所以在头部添加一个元素
            ax.set_yticklabels(y_label)

            plt.show()












