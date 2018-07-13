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
    def show_segment_result(segment_df, date):
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

                cop_scores = np.mean(trial_df[DatabaseInfo.get_cop_column_names()].as_matrix(), axis=1)
                sub_cop_im = Presenter.__get_score_im(cop_scores, axis_1_range, axis_2_range)
                average_cop_im += sub_cop_im
        total_number = SUB_NUM * SPEEDS.__len__()
        average_force_im = average_force_im / total_number
        average_force_title = segment + ', average force'
        Presenter.__show_score_im(average_force_im, axis_1_range, axis_2_range, average_force_title, axis_1_label,
                                  axis_2_label, segment, date)

        average_cop_im = average_cop_im / total_number
        average_cop_title = segment + ', average COP'
        Presenter.__show_score_im(average_cop_im, axis_1_range, axis_2_range, average_cop_title, axis_1_label,
                                  axis_2_label, segment, date)

    # @staticmethod
    # def show_speed_result(speed_df):


    # show the decrease amount and std among subjects for all the speeds
    @staticmethod
    def get_segment_force_matrix(segment_df, sheet=None):
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
        force_title = segment + ', force, mean and one standard deviation of R2 decrease'
        if sheet:
            Presenter.__store_matrix(force_mean_matrix, force_std_matrix, force_title, axis_1_range, axis_2_range, sheet, segment)
        return force_matrix, axis_1_range, axis_2_range

    # show the decrease amount and std among subjects for all the speeds
    @staticmethod
    def get_segment_cop_matrix(segment_df, sheet=None):
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
        cop_matrix = np.zeros([axis_2_len, axis_1_len, total_matrix_number])
        i_matrix = 0
        for i_sub in range(SUB_NUM):
            for speed in SPEEDS:
                trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                center_df = trial_df[(trial_df['x_offset'] == 0) & (trial_df['y_offset'] == 0) &
                                     (trial_df['z_offset'] == 0) & (trial_df['theta_offset'] == 0)]
                cop_center_score = np.mean(center_df[DatabaseInfo.get_cop_column_names()].as_matrix())
                cop_scores = np.mean(trial_df[DatabaseInfo.get_cop_column_names()].as_matrix(), axis=1)
                cop_matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(cop_scores, cop_center_score, axis_1_range, axis_2_range)
                i_matrix += 1
        cop_mean_matrix = np.mean(cop_matrix, axis=2)
        cop_std_matrix = np.std(cop_matrix, axis=2)
        cop_title = segment + ', cop, mean and one standard deviation of R2 decrease'
        if sheet:
            Presenter.__store_matrix(cop_mean_matrix, cop_std_matrix, cop_title, axis_1_range, axis_2_range, sheet, segment)
        return cop_matrix, axis_1_range, axis_2_range

    @staticmethod
    def independence_analysis(segment_df):
        force_matrix, axis_1_range, axis_2_range = Presenter.get_segment_force_matrix(segment_df)
        cop_matrix, axis_1_range, axis_2_range = Presenter.get_segment_cop_matrix(segment_df)
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()

        force_matrix_combined = Presenter.combine_two_axis(force_matrix)
        cop_matrix_combined = Presenter.combine_two_axis(cop_matrix)

    @staticmethod
    def combine_two_axis(original_matrix):
        combined_matrix = original_matrix.copy()
        center_row = int(original_matrix.shape[0] / 2)
        center_col = int(original_matrix.shape[1] / 2)
        for i_row in range(original_matrix.shape[0]):
            for i_col in range(original_matrix.shape[1]):
                combined_matrix[i_row, i_col, :] = original_matrix[i_row, center_col, :] + original_matrix[center_row, i_col, :]
        return combined_matrix


    @staticmethod
    def __store_matrix(mean_matrix, std_matrix, title, axis_1_range, axis_2_range, sheet, segment):
        text_style = xlwt.easyxf('font: name Times New Roman, bold on; align: vert centre, horiz center')
        text_style_2 = xlwt.easyxf('font: name Times New Roman, bold on; align: rotation 90, vert centre, horiz center')
        num_style = xlwt.easyxf('font: name Times New Roman')
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        row_offset = sheet.rows.__len__()
        sheet.write(row_offset, 0, '', text_style)      # 空一行
        row_offset += 1
        # write the title of a matrix
        sheet.write_merge(row_offset, row_offset, 0, 12, title, text_style)
        # write the axis 1 name of a matrix
        if segment in ['trunk', 'pelvis']:
            text = 'axis 1: x from left to right'
        else:
            text = 'axis 1: theta around the leg'
        sheet.write_merge(row_offset+1, row_offset+1, 2, 12, text, text_style)
        # write the axis 2 name of a matrix
        sheet.write_merge(row_offset+3, row_offset+13, 0, 0, 'axis 2: z from up to down', text_style_2)
        # write the label of the first dimension
        i_x = 0
        for axis_1 in axis_1_range:
            if segment in ['trunk', 'pelvis']:
                text = str(axis_1) + 'mm'
            else:
                text = str(axis_1) + '°'
            sheet.write(row_offset+2, i_x+2, text, num_style)
            i_x += 1
        # write the label of the second dimension
        i_y = 0
        for axis_2 in axis_2_range:
            sheet.write(i_y+row_offset+3, 1, str(axis_2)+'mm', num_style)
            i_y += 1

        for i_x in range(axis_1_len):
            for i_y in range(axis_2_len):
                text = str(round(mean_matrix[i_y, i_x], 1)) + ' ± ' + str(round(std_matrix[i_y, i_x], 1))
                sheet.write(i_y+3+row_offset, i_x+2, text, num_style)


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
    def __show_score_im(score_im, axis_1_range, axis_2_range, title, axis_1_label, axis_2_label, segment, date):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = plt.imshow(score_im, cmap=plt.cm.gray)
        plt.colorbar(im)
        if segment in ['trunk', 'pelvis']:
            x_label = [str(tick)+'mm' for tick in axis_1_range]
        else:
            x_label = [str(tick)+'°' for tick in axis_1_range]
        ax.set_xticks(range(score_im.shape[1]))
        ax.set_xticklabels(x_label, fontdict={'fontsize': 8})
        ax.set_xlabel(axis_1_label, fontdict={'fontsize': 12})
        y_label = [str(tick)+'mm' for tick in axis_2_range]
        ax.set_yticks(range(score_im.shape[0]))
        ax.set_yticklabels(y_label, fontdict={'fontsize': 8})
        ax.set_ylabel(axis_2_label, fontdict={'fontsize': 12})
        plt.title(title)
        file_path = RESULT_PATH + 'result_' + date + '\\' + title + '.png'
        plt.savefig(file_path)
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
                decrease_matrix[i_y, i_x] = (scores[i_score] - center_score) * 100
                i_score += 1
        return decrease_matrix

    @staticmethod
    def show_selected_result(segment_df, date, output_names, speed_names):
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
        average_output_im = np.zeros([axis_2_len, axis_1_len])
        for i_sub in range(SUB_NUM):
            for output in output_names:
                for speed in speed_names:
                    trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                    output_scores = trial_df[output].as_matrix()

                    sub_output_im = Presenter.__get_score_im(output_scores, axis_1_range, axis_2_range)
                    average_output_im += sub_output_im

        total_number = SUB_NUM * speed_names.__len__() * output_names.__len__()
        average_output_im = average_output_im / total_number
        output_names_str = ', '.join(output_names)
        average_output_title = segment + ', average ' + output_names_str
        Presenter.__show_score_im(average_output_im, axis_1_range, axis_2_range, average_output_title, axis_1_label,
                                  axis_2_label, segment, date)

    # show the decrease amount and std among subjects for all the speeds
    @staticmethod
    def get_selected_matrix(segment_df, sheet, output_names, speed_names):
        segment = segment_df.iloc[0, 0]
        if segment in ['trunk', 'pelvis']:
            axis_1_range = Presenter.__array_to_range(segment_df['x_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
        else:
            axis_1_range = Presenter.__array_to_range(segment_df['theta_offset'].as_matrix())
            axis_2_range = Presenter.__array_to_range(segment_df['z_offset'].as_matrix())
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        total_matrix_number = SUB_NUM * speed_names.__len__() * output_names.__len__()
        matrix = np.zeros([axis_2_len, axis_1_len, total_matrix_number])
        i_matrix = 0
        for i_sub in range(SUB_NUM):
            for output in output_names:
                for speed in speed_names:
                    trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                    center_df = trial_df[(trial_df['x_offset'] == 0) & (trial_df['y_offset'] == 0) &
                                         (trial_df['z_offset'] == 0) & (trial_df['theta_offset'] == 0)]
                    center_score = np.mean(center_df[output].as_matrix())
                    scores = trial_df[output].as_matrix()
                    matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(scores, center_score,
                                                                                   axis_1_range, axis_2_range)
                    i_matrix += 1
        mean_matrix = np.mean(matrix, axis=2)
        std_matrix = np.std(matrix, axis=2)
        title = segment + ', output = '.join(output_names) + ', speed = ' + ', '.join(speed_names)
        Presenter.__store_matrix(mean_matrix, std_matrix, title, axis_1_range, axis_2_range, sheet, segment)


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
