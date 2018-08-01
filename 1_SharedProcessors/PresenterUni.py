import matplotlib.pyplot as plt
import matplotlib as mpl
import xlwt
from math import sqrt
from DatabaseInfo import DatabaseInfo
from OffsetClass import *
from const import *
import pickle

class Presenter:
    @staticmethod
    def show_segment_result_all(result_name, date):
        with open('segment_result_all.txt', 'rb') as fp:  # Pickling
            result_all = pickle.load(fp)
        if result_name == 'R2':
            im_all = result_all[0]
            bar_title = 'R2 decrease'
            colorbar_tick = np.arange(-0.04, 0.01, 0.01)
        elif result_name == 'RMSE':
            im_all = result_all[1]
            bar_title = 'RMSE increase'
            colorbar_tick = np.arange(0, 0.05, 0.01)
        elif result_name == 'NRMSE':
            im_all = result_all[2]
            bar_title = 'NRMSE increase'
            colorbar_tick = np.arange(0, 1.6, 0.5)
        else:
            raise RuntimeError('Wrong result name')

        # get the max and min of all the imagines
        max_value, min_value = im_all[0][0][0, 0], im_all[0][0][0, 0]
        for segment_im in im_all:
            for im in segment_im:
                max_current = np.max(im)
                min_current = np.min(im)
                if max_current > max_value:
                    max_value = max_current
                if min_current < min_value:
                    min_value = min_current
        x_label_mm = ['-100', '', '', '', '', '0', '', '', '', '', '100']
        label_foot = ['-50', '', '', '', '', '0', '', '', '', '', '50']
        x_label_theta = ['-25', '', '', '', '', '0', '', '', '', '', '25']
        y_label = ['100', '', '', '', '', '0', '', '', '', '', '-100']
        cmap_item = plt.cm.get_cmap('RdYlBu_r')
        # get the first four segments
        fig, ax = plt.subplots(figsize=(7, 8))
        # fig, ax = plt.subplots()
        for i_segment in range(4):
            for i_im in range(len(im_all[i_segment])):
                # plot the foot before shank and thigh
                if i_segment < 2:
                    score_im = im_all[i_segment][i_im]
                else:
                    score_im = im_all[i_segment+4][i_im]
                plt.subplot(4, 3, 3*i_segment+i_im+1)
                ax = plt.gca()
                plt.imshow(score_im, vmin=min_value, vmax=max_value, cmap=cmap_item)     # , cmap=plt.cm.gray
                if i_segment == 1:
                    ax.set_xticks(range(score_im.shape[1]))
                    ax.set_xticklabels(x_label_mm, fontdict={'fontsize': FONT_SIZE})
                elif i_segment == 3:
                    ax.set_xticks(range(score_im.shape[1]))
                    ax.set_xticklabels(label_foot, fontdict={'fontsize': FONT_SIZE})
                else:
                    ax.get_xaxis().set_visible(False)
                if i_im == 0 and i_segment < 2:
                    ax.set_yticks(range(score_im.shape[0]))
                    ax.set_yticklabels(y_label, fontdict={'fontsize': FONT_SIZE})
                elif i_im == 0 and i_segment >= 2:
                    ax.set_yticks(range(score_im.shape[0]))
                    ax.set_yticklabels(label_foot, fontdict={'fontsize': FONT_SIZE})
                else:
                    ax.get_yaxis().set_visible(False)
        plt.text(12, -40, 'trunk', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, -26, 'pelvis', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, -12, 'left foot', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, 1.5, 'right foot', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(-41, -40, 'error in Z (mm)', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(-41, -8, 'error in Y (mm)', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(-18, 15.5, 'error in X (mm)', fontdict={'fontsize': FONT_SIZE})
        plt.text(-31, -46, 'MLGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(-14, -46, 'APGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(3, -46, 'VGRF', fontdict={'fontsize': FONT_SIZE})
        plt.tight_layout(w_pad=1.05, h_pad=1.05)
        fig.subplots_adjust(left=0.15, right=0.92, bottom=0.1, top=0.92)
        file_path = RESULT_PATH + 'result_segment\\' + date + '\\four_segments_1.png'
        plt.savefig(file_path)

        # get the first four segments
        fig, ax = plt.subplots(figsize=(7, 8))
        # fig, ax = plt.subplots()
        for i_segment in range(2, 6):
            for i_im in range(len(im_all[i_segment])):
                # plot the foot before shank and thigh
                score_im = im_all[i_segment][i_im]
                if score_im.shape[1] == 13:     # to make the size the same
                    score_im = score_im[:, 1:-1]
                plt.subplot(4, 3, 3*(i_segment-2)+i_im+1)
                ax = plt.gca()
                plt.imshow(score_im, vmin=min_value, vmax=max_value, cmap=cmap_item)     # , cmap=plt.cm.gray
                if i_segment == 5:
                    ax.set_xticks(range(score_im.shape[1]))
                    ax.set_xticklabels(x_label_theta, fontdict={'fontsize': FONT_SIZE})
                else:
                    ax.get_xaxis().set_visible(False)
                if i_im == 0:
                    ax.set_yticks(range(score_im.shape[0]))
                    ax.set_yticklabels(y_label, fontdict={'fontsize': FONT_SIZE})
                else:
                    ax.get_yaxis().set_visible(False)
        plt.text(12, -36, 'left thigh', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, -24, 'right thigh', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, -11, 'left shank', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(12, 1, 'right shank', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(-36, -20, 'error in Z (mm)', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(-24, 15, 'error in circumference (degree)', fontdict={'fontsize': FONT_SIZE})
        plt.text(-27, -40, 'MLGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(-11.5, -40, 'APGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(3, -40, 'VGRF', fontdict={'fontsize': FONT_SIZE})
        plt.tight_layout(w_pad=1.05, h_pad=1.05)
        fig.subplots_adjust(left=0.14, right=0.92, bottom=0.1, top=0.92)
        file_path = RESULT_PATH + 'result_segment\\' + date + '\\four_segments_2.png'
        plt.savefig(file_path)

        fig = plt.figure(figsize=(1.7, 8))
        ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.83])
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        cb = mpl.colorbar.ColorbarBase(ax1, norm=norm, orientation='vertical', ticks=colorbar_tick, cmap=cmap_item)
        cb.ax.tick_params(labelsize=FONT_SIZE)
        plt.text(3.3, 0.6, bar_title, fontdict={'fontsize': FONT_SIZE}, rotation=90)
        file_path = RESULT_PATH + 'result_segment\\' + date + '\\colorbar.png'
        plt.savefig(file_path)
        plt.show()

    @staticmethod
    def show_segment_result(segment_df, result_names, sub_num=SUB_NUM):
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
        axis_1_range.reverse()
        axis_1_len = axis_1_range.__len__()

        total_number = sub_num * len(SPEEDS)
        result_names_combined = [['FP1.' + name, 'FP2.' + name] for name in result_names]
        segment_im_all = []
        for force in result_names_combined:
            added_im = np.zeros([axis_1_len, axis_0_len])
            for i_sub in range(sub_num):
                for speed in SPEEDS:
                    trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                    sub_im = Presenter.__get_decrease_im_uni(trial_df[[axis_0_name, axis_1_name, force[0],
                                                                       force[1]]], axis_0_range, axis_1_range,
                                                             axis_0_name, axis_1_name)
                    added_im += sub_im

            average_im = added_im / total_number
            segment_im_all.append(average_im)
        return segment_im_all

    @staticmethod
    def __get_axis_name(segment_name):
        if segment_name in ['trunk', 'pelvis']:
            return [segment_name + '_x_offset', segment_name + '_z_offset']
        elif segment_name in ['l_foot', 'r_foot']:
            return [segment_name + '_x_offset', segment_name + '_y_offset']
        else:
            return [segment_name + '_theta_offset', segment_name + '_z_offset']

    @staticmethod
    def show_all_combined_result(result_df, exp1_result, result_names, date, sub_num=SUB_NUM):
        for result_name in result_names:
            average_result_im = np.zeros([16, 16])
            for i_sub in range(sub_num):
                for speed in SPEEDS:
                    exp1_speed_df = exp1_result[
                        (exp1_result['subject_id'] == i_sub) & (exp1_result['speed'] == float(speed))
                        & (exp1_result['segment'] == 'trunk')]
                    original_df_item = exp1_speed_df[(exp1_speed_df['trunk_x_offset'] == 0) &
                                                     (exp1_speed_df['trunk_z_offset'] == 0)]
                    # original_value = original_df_item[result_name].as_matrix()
                    original_value = 5
                    speed_df = result_df[(result_df['subject_id'] == i_sub) & (result_df['speed'] == float(speed))]
                    sub_result_im = Presenter.__get_all_combined_im(speed_df[result_name], original_value)
                    average_result_im += sub_result_im

            total_number = sub_num * len(SPEEDS)
            average_result_im = average_result_im / total_number
            average_force_title = result_name
            Presenter.__show_all_segment_im(average_result_im, average_force_title, date)

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
    def save_segment_result_all(result_df):
        result_all = []
        for target_name in ['R2', 'RMSE', 'NRMSE']:
            result_names = [force_name + '_' + target_name for force_name in ['ForX', 'ForY', 'ForZ']]
            im_all = []
            for segment in SEGMENT_NAMES:
                segment_df = result_df[result_df['segment'] == segment]
                im_all.append(Presenter.show_segment_result(segment_df, result_names))
            result_all.append(im_all)
        with open('segment_result_all.txt', 'wb') as fp:
            pickle.dump(result_all, fp)

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
                score_im[i_y, i_x] = (score[0][-1] + score[0][-2]) / 2
                i_score += 1
        return score_im

    @staticmethod
    def __get_decrease_im_uni(scores_df, axis_0_range, axis_1_range, axis_0_name, axis_1_name):
        axis_0_len = axis_0_range.__len__()
        axis_1_len = axis_1_range.__len__()
        score_im = np.zeros([axis_1_len, axis_0_len])
        center_df = scores_df[(scores_df[axis_0_name] == 0) & (scores_df[axis_1_name] == 0)]
        center_score = np.mean(center_df.as_matrix()[0][2:])
        i_score = 0
        for i_x in range(axis_0_len):
            for i_y in range(axis_1_len):
                # for image, row and column are exchanged compared to ndarray
                score = scores_df[(scores_df[axis_0_name] == axis_0_range[i_x]) &
                                  (scores_df[axis_1_name] == axis_1_range[i_y])].as_matrix()
                score_im[i_y, i_x] = (score[0][-1] + score[0][-2]) / 2 - center_score
                i_score += 1
        return score_im

    @staticmethod
    def __get_all_combined_im(score_df, original_value):
        scores = score_df.as_matrix()
        im_len = 16
        im = np.zeros([im_len, im_len])
        i_score = 0
        for i_x in range(im_len):
            for i_y in range(im_len):
                im[i_y, i_x] = scores[i_score] - original_value
                i_score += 1
        return im


    @staticmethod
    def __get_all_offset_names():
        return ['trunk_x_offset', 'trunk_z_offset',
                'pelvis_x_offset', 'pelvis_z_offset',
                'l_thigh_z_offset', 'l_thigh_theta_offset',
                'r_thigh_z_offset', 'r_thigh_theta_offset',
                'l_shank_z_offset', 'l_shank_theta_offset',
                'r_shank_z_offset', 'r_shank_theta_offset']

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
    def __show_score_im(score_im, axis_0_range, axis_1_range, title, axis_0_label, axis_1_label, folder):

        fig, ax = plt.subplots(figsize=(8, 6))
        im = plt.imshow(score_im, cmap=plt.cm.gray)
        plt.colorbar(im)
        if axis_0_label.__contains__('theta'):
            x_label = [str(int(tick)) + ' °' for tick in axis_0_range]
        else:
            x_label = [str(int(1000 * tick)) + ' mm' for tick in axis_0_range]
        ax.set_xticks(range(score_im.shape[1]))
        ax.set_xticklabels(x_label, fontdict={'fontsize': FONT_SIZE}, rotation=45)
        ax.set_xlabel(axis_0_label, fontdict={'fontsize': FONT_SIZE})
        if axis_1_label.__contains__('theta'):
            y_label = [str(int(tick)) + ' °' for tick in axis_1_range]
        else:
            y_label = [str(int(1000 * tick)) + ' mm' for tick in axis_1_range]
        ax.set_yticks(range(score_im.shape[0]))
        ax.set_yticklabels(y_label, fontdict={'fontsize': FONT_SIZE})
        ax.set_ylabel(axis_1_label, fontdict={'fontsize': FONT_SIZE})
        # plt.title(title, fontdict={'fontsize': FONT_SIZE})
        plt.tight_layout()
        file_path = RESULT_PATH + folder + '\\' + title + '.png'
        plt.savefig(file_path)

    @staticmethod
    def __show_all_segment_im(im, title, date):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = plt.imshow(im, cmap=plt.cm.gray)
        plt.colorbar(im)
        # ax.set_xticks(range(im.shape[1]))
        # ax.set_xticklabels(range(im.shape[1]), fontdict={'fontsize': 8})
        # ax.set_xticks(range(im.shape[0]))
        # ax.set_xticklabels(range(im.shape[0]), fontdict={'fontsize': 8})
        plt.title(title)
        # file_path = RESULT_PATH + 'result_' + date + '\\' + title + '.png'
        # plt.savefig(file_path)
        plt.show()  # show plot at last

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
