import numpy as np
import pandas as pd
from const import *
from SegmentData import SegmentData
from VirtualProcessor import Processor
from EvaluationClass import Evaluation
from sklearn import ensemble
import matplotlib.pyplot as plt


class TrainTestGenerator:

    def __init__(self, subject_data, moved_segment, speed, output_names):
        self.__subject_data = subject_data
        self.__moved_segment = moved_segment
        self.__speed = speed
        # self.__input_names = input_names
        self.__output_names = output_names
        self.__axis_1_range = []
        self.__axis_2_range = []

    def __get_plane_position(self):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        center_point_mean = segment_data.get_center_point_mean(self.__speed)
        # range are in millimeter
        x_range, y_range = [], []
        if self.__moved_segment == 'trunk':
            trunk_limit = 120
            x_range = range(-trunk_limit, trunk_limit+1, 20)
            y_range = range(-trunk_limit, trunk_limit+1, 20)
        elif self.__moved_segment == 'pelvis':
            pelvis_x_limit, pelvis_y_limit = 120, 40
            x_range = range(-pelvis_x_limit, pelvis_x_limit+1, 20)
            y_range = range(-pelvis_y_limit, pelvis_y_limit+1, 10)

        point_list = []
        for x in x_range:
            for y in y_range:
                point = center_point_mean + [x, y, 0]
                point_list.append([point, x, y])     # score, x offset, y offset

        self.__axis_1_range, self.__axis_2_range = x_range, y_range
        return point_list, x_range, y_range

    def __get_cylinder_position(self):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        center_point_mean = segment_data.get_center_point_mean(self.__speed)
        # range are in millimeter
        z_limit, theta_limit = 30, 30       # theta for the degree
        z_range = range(-z_limit, z_limit+1, 10)
        theta_range = range(-theta_limit, theta_limit+1, 10)
        cylinder_diameter = self.__subject_data.get_cylinder_diameter(self.__moved_segment)
        point_list = []
        for theta in theta_range:
            for z in z_range:
                theta_radians = theta * np.pi / 180     # change degree to radians
                R_cylinder = Processor.get_cylinder_surface_rotation(theta_radians)
                x = cylinder_diameter / 2 * (1 - np.cos(theta_radians))
                y = cylinder_diameter / 2 * np.sin(theta_radians)
                point = center_point_mean + [x, y, z]
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
            i_segment += 1
        x = pd.DataFrame(x)
        x.columns = ALL_INPUT_NAMES

        # get y
        y = walking_data_1_df[self.__output_names]

        return x, y

    def get_score_list(self, x_train, x_test, y_train, y_test, do_scaling):
        # score_list = []
        # point_list, x_range, y_range = self.__get_plane_position()
        model_random_forest = ensemble.RandomForestRegressor(random_state=10)
        my_evaluator = Evaluation(x_train, x_test, y_train, y_test, self.__output_names, do_scaling)
        my_evaluator.train_sklearn(model_random_forest)
        score_list = []
        if self.__moved_segment in ['trunk', 'pelvis']:
            score_list = self.get_plane_score_list(my_evaluator, x_test)
        if self.__moved_segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
            score_list = self.get_cylinder_score_list(my_evaluator, x_test)
        return score_list

    def get_plane_score_list(self, evaluator, x_test):
        score_list = []
        point_list, x_range, y_range = self.__get_plane_position()
        for point in point_list:
            simulated_marker = point[0]
            x_test = self.__modify_acc_test(simulated_marker, x_test)
            evaluator.set_x_test(x_test)
            score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
        return score_list

    def get_cylinder_score_list(self, evaluator, x_test):
        score_list = []
        point_list, theta_range, z_range = self.__get_cylinder_position()
        for point in point_list:
            simulated_marker = point[0]
            R_cylinder = point[3]
            x_test = self.__modify_acc_test(simulated_marker, x_test, R_cylinder)
            evaluator.set_x_test(x_test)
            score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
        return score_list

    def __modify_acc_test(self, simulated_marker, x_test, R_cylinder=[]):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
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
        x_test[changed_columns] = acc_IMU_df.loc[test_index]
        return x_test
