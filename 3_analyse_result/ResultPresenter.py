import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt
from xlutils.copy import copy as xl_copy

from const import *
from XYGenerator import XYGenerator
from DatabaseInfo import DatabaseInfo


class Presenter:
    @staticmethod
    def show_segment_result(segment_df):
        segment = segment_df.iloc[0, 0]
        if segment in ['trunk', 'pelvis']:
            axis_1_range = Presenter.__array_to_range(segment_df['x_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
            axis_1_label, axis_2_label = 'x offset to center', 'z offset to center'

        else:
            axis_1_range = Presenter.__array_to_range(segment_df['theta_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
            axis_1_label, axis_2_label = 'theta offset to center', 'z offset to center'

        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        average_force_im = np.zeros([axis_2_len, axis_1_len])
        average_cop_im = np.zeros([axis_2_len, axis_1_len])
        for i_sub in range(SUB_NUM):
            for speed in SPEEDS:
                trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                force_scores = np.mean(trial_df[DatabaseInfo.get_force_column_names()].as_matrix(), axis=1)
                sub_force_im = Presenter.__get_score_im(force_scores, axis_1_range, axis_2_range)
                average_force_im += sub_force_im

                # cop_scores = np.mean(trial_df[DatabaseInfo.get_cop_column_names()].as_matrix(), axis=1)
                # sub_cop_im = Presenter.__get_score_im(cop_scores, axis_1_range, axis_2_range)
                # average_cop_im += sub_cop_im
        total_im_number = SUB_NUM * SPEEDS.__len__()
        average_force_im = average_force_im / total_im_number
        average_force_title = segment + ', average force'
        Presenter.__show_score_im(average_force_im, axis_1_range, axis_2_range, average_force_title, axis_1_label,
                                  axis_2_label, segment)

        # average_cop_im = average_cop_im / total_im_number
        # average_cop_title = segment + ', average COP'
        # Presenter.__show_score_im(average_cop_im, axis_1_range, axis_2_range, average_cop_title)

    # @staticmethod
    # def show_speed_result(speed_df):



    # show the decrease amount and std among subjects for all the speeds
    @staticmethod
    def get_segment_matrix(segment_df, sheet):
        segment = segment_df.iloc[0, 0]
        if segment in ['trunk', 'pelvis']:
            axis_1_range = Presenter.__array_to_range(segment_df['x_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
        else:
            axis_1_range = Presenter.__array_to_range(segment_df['theta_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        total_matrix_number = SUB_NUM * SPEEDS.__len__()
        force_matrix = np.zeros([axis_2_len, axis_1_len, total_matrix_number])
        i_matrix = 0
        for i_sub in range(SUB_NUM):
            for speed in SPEEDS:
                trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                center_df = trial_df[(trial_df['x_offset'] == 0) & (trial_df['y_offset'] == 0) &
                                     (trial_df['z_offset'] == 0) & (trial_df['theta_offset'] == 0)]
                force_center_score = np.mean(center_df[DatabaseInfo.get_force_column_names()].as_matrix())
                force_scores = np.mean(trial_df[DatabaseInfo.get_force_column_names()].as_matrix(), axis=1)
                force_matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(force_scores, force_center_score, axis_1_range, axis_2_range)
                i_matrix += 1
        force_mean_matrix = np.mean(force_matrix, axis=2)
        force_std_matrix = np.std(force_matrix, axis=2)
        force_title = segment + ', mean and one standard deviation of R2 decrease'
        Presenter.__store_matrix(force_mean_matrix, force_std_matrix, force_title, axis_1_range, axis_2_range, sheet, segment)





    @staticmethod
    def __store_matrix(mean_matrix, std_matrix, title, axis_1_range, axis_2_range, sheet, segment):
        text_style = xlwt.easyxf('font: name Times New Roman, bold on')
        num_style = xlwt.easyxf('font: name Times New Roman')
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        row_offset = sheet.rows.__len__()
        # write the title of a matrix
        sheet.write(row_offset, 0, title, text_style)
        # write the label of the first dimension
        i_x = 0
        for axis_1 in axis_1_range:
            if segment in ['trunk', 'pelvis']:
                text = str(axis_1) + 'mm'
            else:
                text = str(axis_1) + '°'
            sheet.write(row_offset+1, i_x+1, text, num_style)
            i_x += 1
        # write the label of the second dimension
        i_y = 0
        for axis_2 in axis_2_range:
            sheet.write(i_y+row_offset+2, 0, str(axis_2)+'mm', num_style)
            i_y += 1

        for i_x in range(axis_1_len):
            for i_y in range(axis_2_len):
                text = str(round(mean_matrix[i_y, i_x], 3)) + ' ± ' + str(round(std_matrix[i_y, i_x], 3))
                sheet.write(i_y+2+row_offset, i_x+1, text, num_style)


    @staticmethod
    def __get_score_im(scores, axis_1_range, axis_2_range):
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        score_im = np.zeros([axis_2_len, axis_1_len])
        i_score = 0
        for i_x in range(axis_1_len):
            for i_y in range(axis_2_len):
                # for image, row and column are exchanged compared to ndarray
                score_im[i_y, i_x] = scores[i_score]
                i_score += 1
        return score_im

    @staticmethod
    def __array_to_range(array):
        start_value = int(array[0])
        end_value = int(np.max(array))
        step_value = 1
        for value in array:
            if value != start_value:
                step_value = int(value - start_value)
                break
        return range(start_value, end_value + 1, step_value)

    # this show result is different from the one in EvaluationClass because it serves analyse result
    @staticmethod
    def __show_score_im(score_im, axis_1_range, axis_2_range, title, axis_1_label, axis_2_label, segment):
        fig, ax = plt.subplots()
        im = plt.imshow(score_im, cmap=plt.cm.gray)
        plt.colorbar(im)
        if segment in ['trunk', 'pelvis']:
            x_label = [str(tick)+'mm' for tick in axis_1_range]
        else:
            x_label = [str(tick)+'°' for tick in axis_1_range]
        ax.set_xticks(range(score_im.shape[1]))
        ax.set_xticklabels(x_label)
        ax.set_xlabel(axis_1_label)
        y_label = [str(tick)+'mm' for tick in axis_2_range]
        ax.set_yticks(range(score_im.shape[0]))
        ax.set_yticklabels(y_label)
        ax.set_ylabel(axis_2_label)
        plt.title(title)
        plt.savefig(segment)
        plt.show()  # show plot at last

    @staticmethod
    def __get_decrease_matrix(scores, center_score, axis_1_range, axis_2_range):
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        decrease_matrix = np.zeros([axis_2_len, axis_1_len])
        i_score = 0
        for i_x in range(axis_1_len):
            for i_y in range(axis_2_len):
                # for image, row and column are exchanged compared to ndarray
                decrease_matrix[i_y, i_x] = scores[i_score] - center_score
                i_score += 1
        return decrease_matrix

    # @staticmethod
    # def __process_decrease_matrix(decrease_matrix):
    #     # axis_2_len, axis_1_len = decrease_matrix.shape[0], decrease_matrix.shape[1]
    #     # total_matrix_number = decrease_matrix.shape[2]
    #     mean_matrix = np.mean()
    #     std_matrix = np.std(decrease_matrix, axis=2)


    # # show the decrease amount and std among subjects for one speed
    # @staticmethod
    # def get_speed_matrix(speed_df):
    #     segment = speed_df.iloc[0, 0]
    #     if segment in ['trunk', 'pelvis']:
    #         axis_1_range = Presenter.__array_to_range(speed_df['x_offset'].as_matrix())
    #         axis_2_range = Presenter.__array_to_range(speed_df['z_offset'].as_matrix())
    #     else:
    #         axis_1_range = Presenter.__array_to_range(speed_df['theta_offset'].as_matrix())
    #         axis_2_range = Presenter.__array_to_range(speed_df['z_offset'].as_matrix())
    #     axis_1_len = axis_1_range.__len__()
    #     axis_2_len = axis_2_range.__len__()
    #     total_matrix_number = SUB_NUM
    #     force_matrix = np.zeros([axis_2_len, axis_1_len, total_matrix_number])
    #     i_matrix = 0
    #     for i_sub in range(SUB_NUM):
    #         trial_df = speed_df[(speed_df['subject_id'] == i_sub)]
    #         center_df = trial_df[(trial_df['x_offset'] == 0) & (trial_df['y_offset'] == 0) &
    #                              (trial_df['z_offset'] == 0) & (trial_df['theta_offset'] == 0)]
    #         force_center_score = np.mean(center_df[DatabaseInfo.get_force_column_names()].as_matrix())
    #         force_scores = np.mean(trial_df[DatabaseInfo.get_force_column_names()].as_matrix(), axis=1)
    #         force_matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(force_scores, force_center_score, axis_1_range, axis_2_range)
    #         i_matrix += 1
    #     force_mean_matrix = np.mean(force_matrix, axis=2)
    #     force_std_matrix = np.std(force_matrix, axis=2)