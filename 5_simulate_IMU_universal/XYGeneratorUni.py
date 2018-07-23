import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from const import *
from SegmentData import SegmentData
from VirtualProcessor import Processor
from EvaluationClass import Evaluation
import matplotlib.pyplot as plt
from OffsetClass import *

class XYGeneratorUni:

    def __init__(self, subject_data, speed, output_names, input_names):
        self.__subject_data = subject_data
        # self.__moved_segment = moved_segment        # 用到这个的都要改掉
        self.__speed = speed
        self.__output_names = output_names
        self.__axis_1_range = []
        self.__axis_2_range = []
        self.__gyr_data = False      # whether to use gyr data
        self.__acc_data = False      # whether to use acc data
        for input_name in input_names:
            if input_name.__contains__('acc'):
                self.__acc_data = True
            if input_name.__contains__('gyr'):
                self.__gyr_data = True

    def get_subject_id(self):
        return self.__subject_data.get_subject_id()

    def get_speed(self):
        return self.__speed



    def get_cylinder_diameter(self, segment):
        if segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
            return self.__subject_data.get_cylinder_diameter(segment)
        return 0        # for all the other segment, return 0

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

    def modify_x(self, x, offset_combo):
        # x needs to be deep copied so that it won't be affected by the modification
        x_changed = x.copy()
        for offset in offset_combo:      # each offset represents one sensor movement
            if self.__acc_data:
                x_changed = self.__modify_acc(x_changed, offset)
            if self.__gyr_data:
                x_changed = self.__modify_gyr(x_changed, offset)
        return x_changed

    def __modify_acc(self, x, offset):
        segment = offset.get_segment()
        segment_data = SegmentData(self.__subject_data, segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
        R_standing_to_ground = segment_data.get_segment_R()
        marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
        simulated_marker = segment_data.get_center_point_mean(self.__speed) + offset.get_translation()
        virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                       marker_cali_matrix, R_standing_to_ground)
        # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
        acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)

        R_cylinder = offset.get_rotation()
        # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
        if R_cylinder is not None:
            acc_IMU = np.matmul(R_cylinder, acc_IMU.T).T

        changed_columns = []
        for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
            column = segment + acc_name
            changed_columns.append(column)

        x[changed_columns] = acc_IMU
        return x

    def __modify_gyr(self, x, offset):
        segment = offset.get_segment()
        segment_data = SegmentData(self.__subject_data, segment)
        walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
        R_standing_to_ground = segment_data.get_segment_R()
        marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
        simulated_marker = segment_data.get_center_point_mean(self.__speed) + offset.get_translation()
        virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
                                                                       marker_cali_matrix, R_standing_to_ground)
        # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
        gyr_IMU = Processor.get_gyr(walking_data_1_df, R_IMU_transform)

        R_cylinder = offset.get_rotation()
        # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
        if R_cylinder is not None:
            gyr_IMU = np.matmul(R_cylinder, gyr_IMU.T).T

        changed_columns = []
        for gyr_name in ['_gyr_x', '_gyr_y', '_gyr_z']:
            column = segment + gyr_name
            changed_columns.append(column)

        x[changed_columns] = gyr_IMU
        return x

    # # this function is used to check how much have the IMU data changed
    # def check_acc_difference(self, x, R_cylinder=[]):
    #     point_list, x_range, z_range = self.__get_plane_position()
    #     for point in point_list:
    #         simulated_marker = point[0]
    #
    #         segment_data = SegmentData(self.__subject_data, self.__moved_segment)
    #         walking_data_1_df = segment_data.get_segment_walking_1_data(self.__speed)
    #         test_index = x.index
    #         R_standing_to_ground = segment_data.get_segment_R()
    #         marker_cali_matrix = segment_data.get_marker_cali_matrix(self.__speed)
    #         virtual_marker, R_IMU_transform = Processor.get_virtual_marker(simulated_marker, walking_data_1_df,
    #                                                                        marker_cali_matrix, R_standing_to_ground)
    #         # Processor.check_virtual_marker(virtual_marker, walking_data_1_df, self.__moved_segment)  # for check
    #         acc_IMU = Processor.get_acc(virtual_marker, R_IMU_transform)
    #
    #         # if it was simulated on cylinder, a rotation around the cylinder surface is necessary
    #         if len(R_cylinder) != 0:
    #             acc_IMU = np.matmul(R_cylinder, acc_IMU.T).T
    #
    #         changed_columns = []
    #         for acc_name in ['_acc_x', '_acc_y', '_acc_z']:
    #             column = self.__moved_segment + acc_name
    #             changed_columns.append(column)
    #         plt.plot(acc_IMU[:, 0])
    #         plt.plot(x[changed_columns].as_matrix()[:, 0])
    #         score = r2_score(acc_IMU[:, 0], x[changed_columns].as_matrix()[:, 0])
    #         plt.title('pelvis, x += ' + str(point[1]) + ', z += ' + str(point[2]) + ', score = ' + str(score))
    #         plt.show()


    # def get_score_list(self, x_train, x_test, y_train, y_test, model):
    #     my_evaluator = Evaluation(x_train, x_test, y_train, y_test, self.__output_names, model)
    #     my_evaluator.train_sklearn()
    #     score_list = []
    #     if self.__moved_segment in ['trunk', 'pelvis']:
    #         score_list = self.get_plane_score_list(my_evaluator, x_train, x_test)
    #     if self.__moved_segment in ['l_thigh', 'r_thigh', 'l_shank', 'r_shank']:
    #         score_list = self.get_cylinder_score_list(my_evaluator, x_train, x_test)
    #     return score_list
    #
    # def get_plane_score_list(self, evaluator, x_train, x_test):
    #     score_list = []
    #     point_list, x_range, z_range = self.__get_plane_position()
    #     for point in point_list:
    #         simulated_marker = point[0]
    #         if self.__acc_data:
    #             x_train_changed, x_test_changed = self.__modify_acc(simulated_marker, x_train, x_test)
    #         if self.__gyr_data:
    #             x_train_changed, x_test_changed = self.__modify_gyr(simulated_marker, x_train, x_test)
    #         evaluator.set_x(x_train_changed, x_test_changed)
    #         score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
    #     return score_list
    #
    # def get_cylinder_score_list(self, evaluator, x_train, x_test):
    #     score_list = []
    #     point_list, theta_range, z_range = self.__get_cylinder_position()
    #     for point in point_list:
    #         simulated_marker = point[0]
    #         R_cylinder = point[3]
    #         if self.__acc_data:
    #             x_train_changed, x_test_changed = self.__modify_acc(simulated_marker, x_train, x_test, R_cylinder)
    #         if self.__gyr_data:
    #             x_train_changed, x_test_changed = self.__modify_gyr(simulated_marker, x_train, x_test, R_cylinder)
    #         evaluator.set_x(x_train_changed, x_test_changed)
    #         score_list.append([evaluator.evaluate_sklearn(), point[1], point[2]])  # score, x offset, y offset
    #     return score_list
