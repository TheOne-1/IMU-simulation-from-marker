import matplotlib.pyplot as plt
import xlwt

from DatabaseInfo import DatabaseInfo
from OffsetClass import *
from const import *


class Presenter:
    @staticmethod
    def show_segment_result(segment_df, date, sub_num=SUB_NUM):
        segment_name = segment_df['segment'].iloc[0]
        axes_names = Presenter.__get_axis_name(segment_name)
        axis_0_name = axes_names[0]
        axis_0_value = segment_df[axis_0_name]
        axis_0_range = list(set(axis_0_value))
        axis_0_range.sort()
        axis_0_len = axis_0_range.__len__()

        axis_1_name = axes_names[1]
        axis_1_value = segment_df[axis_1_name]
        axis_1_range = list(set(axis_1_value))
        axis_1_range.sort()
        axis_1_len = axis_1_range.__len__()

        force_names = DatabaseInfo.get_force_column_names()
        for force in force_names:
            average_force_im = np.zeros([axis_1_len, axis_0_len])
            for i_sub in range(sub_num):
                for speed in SPEEDS:
                    trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                    sub_force_im = Presenter.__get_score_im_uni(trial_df[[axis_0_name, axis_1_name, force]],
                                                            axis_0_range, axis_1_range, axis_0_name, axis_1_name)
                    average_force_im += sub_force_im

            total_number = sub_num * SPEEDS.__len__()
            average_force_im = average_force_im / total_number
            average_force_title = segment_name + ', ' + force
            Presenter.__show_score_im(average_force_im, axis_0_range, axis_1_range, average_force_title, axis_0_name,
                                      axis_1_name, date)

    @staticmethod
    def __get_axis_name(segment_name):
        if segment_name in ['trunk', 'pelvis']:
            return [segment_name + '_x_offset', segment_name + '_z_offset']
        elif segment_name in ['l_foot', 'r_foot']:
            return [segment_name + '_x_offset', segment_name + '_y_offset']
        else:
            return [segment_name + '_theta_offset', segment_name + '_z_offset']

    # @staticmethod
    # def show_result_uni(result_df):
    #     for speed in SPEEDS:
    #         speed_df = result_df[result_df['speed'] == float(speed)].copy()
    #         speed_df = speed_df.reset_index(drop=True)
    #         combo_num = 64
    #         scores = np.zeros([combo_num, 10])
    #         for i_sub in range(SUB_NUM):
    #             sub_df = speed_df[]


    @staticmethod
    def show_result_multi_axis(result_df, axes_names, output):
        if len(axes_names) == 1:
            axis_name = axes_names[0]
            axis_values = result_df[axis_name]
            axis_range = list(set(axis_values))
            axis_range.sort()
            score_im = np.zeros([1, len(axis_range)])
            for i_sub in range(SUB_NUM):
                irrelevant_offset = Presenter.__get_all_offset_names()
                irrelevant_offset.remove(axis_name)
                trial_df = result_df[result_df['subject_id'] == i_sub]
                for offset in irrelevant_offset:
                    trial_df = trial_df[trial_df[offset] == 0]

                scores = trial_df[output].as_matrix()
                sub_score_im = Presenter.__get_score_im(scores, axis_range, range(1))
                score_im += sub_score_im
            im_len = SUB_NUM
            score_im /= im_len

        if len(axes_names) == 2:
            axis_0_name = axes_names[0]
            axis_values_0 = result_df[axis_0_name]
            axis_0_range = list(set(axis_values_0))
            axis_0_range.sort()

            axis_1_name = axes_names[1]
            axis_values_1 = result_df[axis_1_name]
            axis_1_range = list(set(axis_values_1))
            axis_1_range.sort()
            score_im = np.zeros([len(axis_1_range), len(axis_0_range)])
            for i_sub in range(SUB_NUM):
                irrelevant_offset = Presenter.__get_all_offset_names()
                irrelevant_offset.remove(axis_0_name)
                irrelevant_offset.remove(axis_1_name)
                trial_df = result_df[result_df['subject_id'] == i_sub]
                for offset in irrelevant_offset:
                    trial_df = trial_df[trial_df[offset] == 0]

                sub_score_im = Presenter.__get_score_im_uni(trial_df[[axis_0_name, axis_1_name, output]],
                                                            axis_0_range, axis_1_range, axis_0_name, axis_1_name)
                score_im += sub_score_im
            im_len = SUB_NUM
            score_im /= im_len
            Presenter.__show_score_im(score_im, axis_0_range, axis_1_range, 'title', axis_0_name, axis_1_name, '77')

    @staticmethod
    def __get_score_im_uni(scores_df, axis_0_range, axis_1_range, axis_0_name, axis_1_name):
        axis_0_len = axis_0_range.__len__()
        axis_1_len = axis_1_range.__len__()
        score_im = np.zeros([axis_1_len, axis_0_len])
        i_score = 0
        for i_x in range(axis_0_len):
            for i_y in range(axis_1_len):
                # for image, row and column are exchanged compared to ndarray
                score = scores_df[(scores_df[axis_0_name] == axis_0_range[i_x]) &
                                  (scores_df[axis_1_name] == axis_1_range[i_y])].as_matrix()
                score_im[i_y, i_x] = score[0][-1]
                i_score += 1
        return score_im

    @staticmethod
    def __get_all_offset_names():
        return ['trunk_x_offset', 'trunk_z_offset',
                'pelvis_x_offset', 'pelvis_z_offset',
                'l_thigh_z_offset', 'l_thigh_theta_offset',
                'r_thigh_z_offset', 'r_thigh_theta_offset',
                'l_shank_z_offset', 'l_shank_theta_offset',
                'r_shank_z_offset', 'r_shank_theta_offset']

    # ********************************* matrix ************************************** #
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
                force_matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(force_scores, force_center_score,
                                                                               axis_1_range, axis_2_range)
                i_matrix += 1
        force_mean_matrix = np.mean(force_matrix, axis=2)
        force_std_matrix = np.std(force_matrix, axis=2)
        force_title = segment + ', force, mean and one standard deviation of R2 decrease'
        if sheet:
            Presenter.__store_matrix(force_mean_matrix, force_std_matrix, force_title, axis_1_range, axis_2_range,
                                     sheet,
                                     segment)
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
                cop_matrix[:, :, i_matrix] = Presenter.__get_decrease_matrix(cop_scores, cop_center_score, axis_1_range,
                                                                             axis_2_range)
                i_matrix += 1
        cop_mean_matrix = np.mean(cop_matrix, axis=2)
        cop_std_matrix = np.std(cop_matrix, axis=2)
        cop_title = segment + ', cop, mean and one standard deviation of R2 decrease'
        if sheet:
            Presenter.__store_matrix(cop_mean_matrix, cop_std_matrix, cop_title, axis_1_range, axis_2_range, sheet,
                                     segment)
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
                combined_matrix[i_row, i_col, :] = original_matrix[i_row, center_col, :] + original_matrix[center_row,
                                                                                           i_col, :]
        return combined_matrix

    @staticmethod
    def __store_matrix(mean_matrix, std_matrix, title, axis_1_range, axis_2_range, sheet, segment):
        text_style = xlwt.easyxf('font: name Times New Roman, bold on; align: vert centre, horiz center')
        text_style_2 = xlwt.easyxf('font: name Times New Roman, bold on; align: rotation 90, vert centre, horiz center')
        num_style = xlwt.easyxf('font: name Times New Roman')
        axis_1_len = axis_1_range.__len__()
        axis_2_len = axis_2_range.__len__()
        row_offset = sheet.rows.__len__()
        sheet.write(row_offset, 0, '', text_style)  # 空一行
        row_offset += 1
        # write the title of a matrix
        sheet.write_merge(row_offset, row_offset, 0, 12, title, text_style)
        # write the axis 1 name of a matrix
        if segment in ['trunk', 'pelvis']:
            text = 'axis 1: x from left to right'
        else:
            text = 'axis 1: theta around the leg'
        sheet.write_merge(row_offset + 1, row_offset + 1, 2, 12, text, text_style)
        # write the axis 2 name of a matrix
        sheet.write_merge(row_offset + 3, row_offset + 13, 0, 0, 'axis 2: z from up to down', text_style_2)
        # write the label of the first dimension
        i_x = 0
        for axis_1 in axis_1_range:
            if segment in ['trunk', 'pelvis']:
                text = str(axis_1) + 'mm'
            else:
                text = str(axis_1) + '°'
            sheet.write(row_offset + 2, i_x + 2, text, num_style)
            i_x += 1
        # write the label of the second dimension
        i_y = 0
        for axis_2 in axis_2_range:
            sheet.write(i_y + row_offset + 3, 1, str(axis_2) + 'mm', num_style)
            i_y += 1

        for i_x in range(axis_1_len):
            for i_y in range(axis_2_len):
                text = str(round(mean_matrix[i_y, i_x], 1)) + ' ± ' + str(round(std_matrix[i_y, i_x], 1))
                sheet.write(i_y + 3 + row_offset, i_x + 2, text, num_style)

    @staticmethod
    def __get_score_im(scores, axis_0_range, axis_1_range):
        axis_0_len = axis_0_range.__len__()
        axis_1_len = axis_1_range.__len__()
        score_im = np.zeros([axis_1_len, axis_0_len])
        i_score = 0
        for i_x in range(axis_0_len):
            for i_y in range(axis_1_len):
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
    def __show_score_im(score_im, axis_0_range, axis_1_range, title, axis_0_label, axis_1_label, date):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = plt.imshow(score_im, cmap=plt.cm.gray)
        plt.colorbar(im)
        if axis_0_label.__contains__('theta'):
            x_label = [str(tick) + '°' for tick in axis_0_range]
        else:
            x_label = [str(1000 * tick) + 'mm' for tick in axis_0_range]
        ax.set_xticks(range(score_im.shape[1]))
        ax.set_xticklabels(x_label, fontdict={'fontsize': 8})
        ax.set_xlabel(axis_0_label, fontdict={'fontsize': 12})
        if axis_1_label.__contains__('theta'):
            y_label = [str(tick) + '°' for tick in axis_1_range]
        else:
            y_label = [str(1000 * tick) + 'mm' for tick in axis_1_range]
        ax.set_yticks(range(score_im.shape[0]))
        ax.set_yticklabels(y_label, fontdict={'fontsize': 8})
        ax.set_ylabel(axis_1_label, fontdict={'fontsize': 12})
        plt.title(title)
        # file_path = RESULT_PATH + 'result_' + date + '\\' + title + '.png'
        # plt.savefig(file_path)
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
                                  axis_2_label, date)

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
