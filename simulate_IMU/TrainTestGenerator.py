import numpy as np
import pandas as pd
from const import *
from SegmentData import SegmentData
from VirtualProcessor import Processor
from EvaluationClass import Evaluation
from sklearn import ensemble
import matplotlib.pyplot as plt


class TrainTestGenerator:

    def __init__(self, subject_data, moved_segment, speed, input_names, output_names):
        self.__subject_data = subject_data
        self.__moved_segment = moved_segment
        self.__speed = speed
        self.__input_names = input_names
        self.__output_names = output_names
        self.__point_list, self.__x_range, self.__y_range = self.__get_simulate_position()

    def __get_simulate_position(self):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        center_point_mean = segment_data.get_center_point_mean(self.__speed)
        # the point list that will generate virtual marker
        point_list, x_range, y_range = self.__get_surface_points(center_point_mean, self.__moved_segment)
        return point_list, x_range, y_range

    def get_point_list(self):
        return self.__point_list

    def get_point_range(self):
        return self.__x_range, self.__y_range

    def get_moved_segment(self):
        return self.__moved_segment

    def __get_surface_points(self, center_point, segment):
        if segment in ['trunk', 'pelvis']:
            return TrainTestGenerator.__get_plane_surface_points(center_point, segment)
        elif segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
            return self.__get_cylinder_surface(center_point, segment)

    # plane surface for trunk and pelvis
    @staticmethod
    def __get_plane_surface_points(center_point, segment):
        # range are in millimeter
        x_range, y_range = [], []
        if segment == 'trunk':
            trunk_limit = 80
            x_range = range(-trunk_limit, trunk_limit+1, 20)
            y_range = range(-trunk_limit, trunk_limit+1, 20)
        elif segment == 'pelvis':
            pelvis_x_limit = 20
            pelvis_y_limit = 5
            x_range = range(-pelvis_x_limit, pelvis_x_limit+1, 10)
            y_range = range(-pelvis_y_limit, pelvis_y_limit+1, 5)

        point_list = []
        for x in x_range:
            for y in y_range:
                point = center_point + [x, y, 0]
                point_list.append([point, x, y])     # score, x offset, y offset
        return point_list, x_range, y_range

    # cylinder surface for thigh and shank
    def __get_cylinder_surface(self, center_point, segment):
        # range are in millimeter
        z_limit, y_limit = 100, 50
        z_range = range(-z_limit, z_limit+1, 10)
        y_range = range(-y_limit, y_limit+1, 10)
        cylinder_diameter = self.__subject_data.get_cylinder_diameter(segment)

        point_list = []
        for z in z_range:
            for y in y_range:
                x = - np.sqrt((cylinder_diameter/2)**2-y**2)
                plt.plot(x, y)      # !! for test
                plt.show()      # !! for test
                point = center_point + [x, y, z]
                point_list.append(point)
        return point_list, y_range, z_range

    # get train and test data
    def get_xy(self):
        speed = self.__speed
        walking_data_1_df = self.__subject_data.get_walking_1_data(speed)
        # for now we only use walking data 1
        data_len = walking_data_1_df.shape[0]
        input_len = self.__input_names.__len__()

        # get x
        x = np.zeros([data_len, input_len])
        i_segment = 0
        for segment_name in SEGMENT_NAMES:
            segment_data = SegmentData(self.__subject_data, segment_name)
            segment_data_walking_1_df = segment_data.get_segment_walking_1_data(speed)
            center_marker = segment_data.get_center_point_mean(speed)
            R_standing_to_ground = segment_data.get_segment_R()
            marker_cali_matrix = segment_data.get_marker_cali_matrix(speed)
            virtual_marker, R_IMU_transform = Processor.get_virtual_marker(center_marker, segment_data_walking_1_df,
                                                                           marker_cali_matrix, R_standing_to_ground)
            # self.check_virtual_marker(virtual_marker, walking_data_1_df)    # for check
            acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)
            x[:, 3*i_segment:3*(i_segment+1)] = acc_IMU
            i_segment += 1
        x = pd.DataFrame(x)
        x.columns = self.__input_names

        # get y
        y = walking_data_1_df[self.__output_names]

        return x, y

    def get_score_list(self, x_train, x_test, y_train, y_test):
        score_list = []
        point_list, x_range, y_range = self.__get_simulate_position()
        for point in point_list:
            simulated_marker = point[0]
            x_test = self.__modify_acc_test(simulated_marker, x_test)
            my_evaluator = Evaluation(x_train, x_test, y_train, y_test, self.__output_names)
            model_random_forest = ensemble.RandomForestRegressor(random_state=10)
            my_evaluator.train_sklearn(model_random_forest)
            score_list.append([my_evaluator.evaluate_sklearn(), point[1], point[2]])     # score, x offset, y offset
        return score_list

    def __modify_acc_test(self, simulated_marker, x_test):
        segment_data = SegmentData(self.__subject_data, self.__moved_segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
        test_index = x_test.index
        R_standing_to_ground = segment_data.get_segment_R()
        marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
        virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                       marker_cali_matrix, R_standing_to_ground)
        acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)

        changed_columns = []
        for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
            column = self.__moved_segment + acc_name
            changed_columns.append(column)

        acc_IMU_df = pd.DataFrame(acc_IMU)
        temp = acc_IMU_df.loc[test_index]
        x_test[changed_columns] = temp
        return x_test
