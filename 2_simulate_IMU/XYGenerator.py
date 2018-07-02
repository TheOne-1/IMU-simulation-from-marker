import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from const import *
from SegmentData import SegmentData
from VirtualProcessor import Processor
from EvaluationClass import Evaluation
import matplotlib.pyplot as plt


class XYGenerator:

    def __init__(self, subject_data, moved_segment, speed, output_names, gyr_data=False, acc_data=True):
        self.__subject_data = subject_data
        self.__moved_segment = moved_segment
        self.__speed = speed
        self.__output_names = output_names
        self.__gyr_data = gyr_data      # whether to use gyr data
        self.__acc_data = acc_data      # whether to use acc data
        self.__axis_1_range = []
        self.__axis_2_range = []

    def get_subject_id(self):
        return self.__subject_data.get_subject_id()

    def get_speed(self):
        return self.__speed

    @staticmethod
    def get_move_range(segment):
        if segment == 'trunk':
            trunk_limit = 100
            x_range = range(-trunk_limit, trunk_limit+1, 20)
            z_range = range(-trunk_limit, trunk_limit+1, 20)
            return x_range, z_range
        elif segment == 'pelvis':
            pelvis_x_limit, pelvis_y_limit = 100, 100
            x_range = range(-pelvis_x_limit, pelvis_x_limit+1, 20)
            z_range = range(-pelvis_y_limit, pelvis_y_limit+1, 20)
            return x_range, z_range
        else:
            theta_limit, z_limit = 30, 100  # theta for the degree
            theta_range = range(-theta_limit, theta_limit + 1, 5)
            z_range = range(-z_limit, z_limit + 1, 20)
            return theta_range, z_range

    def __get_plane_position(self):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        center_point_mean = segment_data.get_center_point_mean(self.__speed)
        # range are in millimeter
        x_range, z_range = XYGenerator.get_move_range(self.__moved_segment)
        point_list = []
        for x in x_range:
            for z in z_range:
                point = center_point_mean + np.array([x, 0, z]) / 1000      # change milliter to meter
                point_list.append([point, x, z])     # score, x offset, y offset

        self.__axis_1_range, self.__axis_2_range = x_range, z_range
        return point_list, x_range, z_range

    def get_cylinder_diameter(self):
        if self.__moved_segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
            return self.__subject_data.get_cylinder_diameter(self.__moved_segment)
        return 0        # for all the other segment, return 0

    def __get_cylinder_position(self):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        center_point_mean = segment_data.get_center_point_mean(self.__speed)
        # range are in millimeter
        theta_range, z_range = XYGenerator.get_move_range(self.__moved_segment)
        cylinder_diameter = self.__subject_data.get_cylinder_diameter(self.__moved_segment)
        point_list = []
        for theta in theta_range:
            for z in z_range:
                theta_radians = theta * np.pi / 180     # change degree to radians
                R_cylinder = Processor.get_cylinder_surface_rotation(theta_radians)
                x = cylinder_diameter / 2 * (1 - np.cos(theta_radians))
                y = cylinder_diameter / 2 * np.sin(theta_radians)
                point = center_point_mean + np.array([x, y, z]) / 1000      # change milliter to meter
                point_list.append([point, theta, z, R_cylinder])

        self.__axis_1_range, self.__axis_2_range = theta_range, z_range
        return point_list, theta_range, z_range

    def get_point_range(self):
        return self.__axis_1_range, self.__axis_2_range

    def get_moved_segment(self):
        return self.__moved_segment

    # get train and test data
    def get_xy(self):
        speed = self.__speed
        walking_data_1_df = self.__subject_data.get_walking_1_data(speed)
        # for now we only use walking data 1
        data_len = walking_data_1_df.shape[0]
        # get x
        if self.__gyr_data:
            x = np.zeros([data_len, 48])
        else:
            x = np.zeros([data_len, 24])
        i_segment = 0
        for segment_name in SEGMENT_NAMES:
            segment_data = SegmentData(self.__subject_data, segment_name)
            segment_data_walking_1_df = segment_data.get_segment_walking_1_data(speed)
            center_marker = segment_data.get_center_point_mean(speed)
            R_standing_to_ground = segment_data.get_segment_R()
            marker_cali_matrix = segment_data.get_marker_cali_matrix(speed)
            virtual_marker, R_IMU_transform = Processor.get_virtual_marker(center_marker, segment_data_walking_1_df,
                                                                           marker_cali_matrix, R_standing_to_ground)
            # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, segment_name)    # for check
            acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)
            x[:, 3*i_segment:3*(i_segment+1)] = acc_IMU

            if self.__gyr_data:
                gyr_IMU = Processor.get_gyr(segment_data_walking_1_df, R_IMU_transform)
                x[:, 24+3*i_segment:24+3*(i_segment+1)] = gyr_IMU

            i_segment += 1
        x = pd.DataFrame(x)
        if self.__gyr_data:
            x.columns = ALL_ACC_GYR_NAMES
        else:
            x.columns = ALL_ACC_NAMES

        # get y
        y = walking_data_1_df[self.__output_names]

        return x, y

    def get_score_list(self, x_train, x_test, y_train, y_test, model):
        my_evaluator = Evaluation(x_train, x_test, y_train, y_test, self.__output_names, model)
        my_evaluator.train_sklearn()
        score_list = []
        if self.__moved_segment in ['trunk', 'pelvis']:
            score_list = self.get_plane_score_list(my_evaluator, x_train, x_test)
        if self.__moved_segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
            score_list = self.get_cylinder_score_list(my_evaluator, x_train, x_test)
        return score_list

    def get_plane_score_list(self, evaluator, x_train, x_test):
        score_list = []
        point_list, x_range, z_range = self.__get_plane_position()
        for point in point_list:
            simulated_marker = point[0]
            if self.__acc_data:
                x_train_changed, x_test_changed = self.__modify_acc_test(simulated_marker, x_train, x_test)
            if self.__gyr_data:
                x_train_changed, x_test_changed = self.__modify_gyr_test(simulated_marker, x_train, x_test)
            evaluator.set_x(x_train_changed, x_test_changed)
            score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
        return score_list

    def get_cylinder_score_list(self, evaluator, x_train, x_test):
        score_list = []
        point_list, theta_range, z_range = self.__get_cylinder_position()
        for point in point_list:
            simulated_marker = point[0]
            R_cylinder = point[3]
            if self.__acc_data:
                x_train_changed, x_test_changed = self.__modify_acc_test(simulated_marker, x_train, x_test, R_cylinder)
            if self.__gyr_data:
                x_train_changed, x_test_changed = self.__modify_gyr_test(simulated_marker, x_train, x_test, R_cylinder)
            evaluator.set_x(x_train_changed, x_test_changed)
            score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
        return score_list

    def __modify_acc_test(self, simulated_marker, x_train, x_test, R_cylinder=[]):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
        train_index = x_train.index
        test_index = x_test.index
        R_standing_to_ground = segment_data.get_segment_R()
        marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
        virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                       marker_cali_matrix, R_standing_to_ground)
        # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
        acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)

        # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
        if len(R_cylinder) != 0:
            acc_IMU = np.matmul(R_cylinder, acc_IMU.T).T

        changed_columns = []
        for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
            column = self.__moved_segment + acc_name
            changed_columns.append(column)

        acc_IMU_df = pd.DataFrame(acc_IMU)
        # x_test is just a quote of the original x_test, deep copy so that no warning will show up
        x_train_changed = x_train.copy()
        x_test_changed = x_test.copy()
        x_train_changed[changed_columns] = acc_IMU_df.loc[train_index]
        x_test_changed[changed_columns] = acc_IMU_df.loc[test_index]
        return x_train_changed, x_test_changed

    def __modify_gyr_test(self, simulated_marker, x_train, x_test, R_cylinder=[]):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
        train_index = x_train.index
        test_index = x_test.index
        R_standing_to_ground = segment_data.get_segment_R()
        marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
        virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                       marker_cali_matrix, R_standing_to_ground)
        # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
        gyr_IMU = Processor.get_gyr(walking_data_1_df, R_IMU_transform)

        # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
        if len(R_cylinder) != 0:
            gyr_IMU = np.matmul(R_cylinder, gyr_IMU.T).T

        changed_columns = []
        for gyr_name in ['_gyr_x', '_gyr_y', '_gyr_z']:
            column = self.__moved_segment + gyr_name
            changed_columns.append(column)

        gyr_IMU_df = pd.DataFrame(gyr_IMU)
        # x_test is just a quote of the original x_test, deep copy so that no warning will show up
        x_train_changed = x_train.copy()
        x_test_changed = x_test.copy()
        x_train_changed[changed_columns] = gyr_IMU_df.loc[train_index]
        x_test_changed[changed_columns] = gyr_IMU_df.loc[test_index]
        return x_train_changed, x_test_changed

    # this function is used to check how much have the IMU data changed
    def check_acc_difference(self, x, R_cylinder=[]):
        point_list, x_range, z_range = self.__get_plane_position()
        for point in point_list:
            simulated_marker = point[0]

            segment_data = SegmentData(self.__subject_data, self.__moved_segment)
            walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
            test_index = x.index
            R_standing_to_ground = segment_data.get_segment_R()
            marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
            virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                           marker_cali_matrix, R_standing_to_ground)
            # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
            acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)

            # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
            if len(R_cylinder) != 0:
                acc_IMU = np.matmul(R_cylinder, acc_IMU.T).T

            changed_columns = []
            for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
                column = self.__moved_segment + acc_name
                changed_columns.append(column)
            plt.plot(acc_IMU[:, 0])
            plt.plot(x[changed_columns].as_matrix()[:, 0])
            score = r2_score(acc_IMU[:, 0], x[changed_columns].as_matrix()[:, 0])
            plt.title('pelvis, x += ' + str(point[1]) + ', z += ' + str(point[2]) + ', score = ' + str(score))
            plt.show()
