import pickle

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import xlwt

from DatabaseInfo import DatabaseInfo
from OffsetClass import *
from const import *


class Presenter:
    @staticmethod
    def show_segment_translation_result(result_name, date):
        with open(RESULT_PATH + 'result_segment_translation\\' + date + '\\segment_result_all.txt', 'rb') as fp:
            result_all = pickle.load(fp)
        if result_name == 'R2':
            im_all = result_all[0]
            bar_title = 'R2 decrease'
            colorbar_tick = np.arange(-0.04, 0.01, 0.01)
        elif result_name == 'RMSE':
            im_all = result_all[1]
            bar_title = 'RMSE increase (BW)'
            colorbar_tick = np.arange(0, 0.05, 0.01)
        elif result_name == 'NRMSE':
            im_all = result_all[2]
            bar_title = 'NRMSE increase (%)'
            colorbar_tick = np.arange(0, 1.1, 0.2)
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
                    score_im = im_all[i_segment + 4][i_im]
                plt.subplot(4, 3, 3 * i_segment + i_im + 1)
                ax = plt.gca()
                plt.imshow(score_im, vmin=min_value, vmax=max_value, cmap=cmap_item)  # , cmap=plt.cm.gray
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
        file_path = RESULT_PATH + 'result_segment_translation\\' + date + '\\four_segments_1.png'
        plt.savefig(file_path)

        # get the first four segments
        fig, ax = plt.subplots(figsize=(7, 8))
        # fig, ax = plt.subplots()
        for i_segment in range(2, 6):
            for i_im in range(len(im_all[i_segment])):
                # plot the foot before shank and thigh
                score_im = im_all[i_segment][i_im]
                if score_im.shape[1] == 13:  # to be compatible to old protocol by making the size the same
                    score_im = score_im[:, 1:-1]
                plt.subplot(4, 3, 3 * (i_segment - 2) + i_im + 1)
                ax = plt.gca()
                plt.imshow(score_im, vmin=min_value, vmax=max_value, cmap=cmap_item)  # , cmap=plt.cm.gray
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
        file_path = RESULT_PATH + 'result_segment_translation\\' + date + '\\four_segments_2.png'
        plt.savefig(file_path)

        fig = plt.figure(figsize=(1.7, 8))
        ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.83])
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        cb = mpl.colorbar.ColorbarBase(ax1, norm=norm, orientation='vertical', ticks=colorbar_tick, cmap=cmap_item)
        cb.ax.tick_params(labelsize=FONT_SIZE)
        plt.text(3.3, 0.6, bar_title, fontdict={'fontsize': FONT_SIZE}, rotation=90)
        file_path = RESULT_PATH + 'result_segment_translation\\' + date + '\\colorbar.png'
        plt.savefig(file_path)
        plt.show()

    @staticmethod
    def get_acceptable_area(result_name, date, threshold=0.5):
        with open(RESULT_PATH + 'result_segment\\' + date + '\\segment_result_all.txt', 'rb') as fp:  # Pickling
            result_all = pickle.load(fp)
        if result_name == 'R2':
            im_all = result_all[0]
        elif result_name == 'RMSE':
            im_all = result_all[1]
        elif result_name == 'NRMSE':
            im_all = result_all[2]
        else:
            raise RuntimeError('Wrong result name')

        area_all = []
        for i_segment in range(len(SEGMENT_NAMES)):
            segment_im = im_all[i_segment]
            axis_0_len = segment_im[0].shape[1]
            axis_1_len = segment_im[0].shape[0]
            segment_acceptable_area = np.zeros(segment_im[0].shape)
            for i_x in range(axis_0_len):
                for i_y in range(axis_1_len):
                    acceptable_flag = True
                    for score_im in segment_im:
                        if score_im[i_y, i_x] > threshold:
                            acceptable_flag = False
                    if acceptable_flag:
                        segment_acceptable_area[i_y, i_x] = 1
            area_all.append(segment_acceptable_area)

        x_label_mm = ['-100', '', '', '', '', '0', '', '', '', '', '100']
        label_foot = ['-50', '', '', '', '', '0', '', '', '', '', '50']
        x_label_theta = ['-25', '', '', '', '', '0', '', '', '', '', '25']
        y_label = ['100', '', '', '', '', '0', '', '', '', '', '-100']
        cmap_item = plt.cm.get_cmap('RdYlGn')
        # get the first four segments
        fig, ax = plt.subplots(figsize=(6.2, 9))
        for i_segment in range(len(SEGMENT_NAMES)):
            segment_acceptable_area = area_all[i_segment]
            # plot the foot before shank and thigh
            if i_segment < 2:
                plt.subplot(4, 2, i_segment * 2 + 1)
            elif 2 <= i_segment < 6:
                plt.subplot(4, 2, i_segment * 2 - 2)
            else:
                plt.subplot(4, 2, i_segment * 2 - 7)
            plt.imshow(segment_acceptable_area, vmax=1, vmin=0, cmap=cmap_item)  # , cmap=plt.cm.gray
            plt.title(SEGMENT_OFFICIAL_NAMES[i_segment], fontdict={'fontsize': FONT_SIZE})
            ax = plt.gca()
            if i_segment < 2:
                ax.set_yticks(range(segment_acceptable_area.shape[0]))
                ax.set_yticklabels(y_label, fontdict={'fontsize': FONT_SIZE})
                if i_segment == 1:
                    ax.set_xticks(range(segment_acceptable_area.shape[1]))
                    ax.set_xticklabels(x_label_mm, fontdict=FONT_DICT)
                else:
                    ax.get_xaxis().set_visible(False)
            elif 2 <= i_segment < 6:
                ax.set_yticks(range(segment_acceptable_area.shape[0]))
                ax.set_yticklabels(y_label, fontdict=FONT_DICT)
                if i_segment == 5:
                    ax.set_xticks(range(segment_acceptable_area.shape[1]))
                    ax.set_xticklabels(x_label_theta, fontdict=FONT_DICT)
                else:
                    ax.get_xaxis().set_visible(False)
            else:
                ax.set_yticks(range(segment_acceptable_area.shape[0]))
                ax.set_yticklabels(label_foot, fontdict=FONT_DICT)
                if i_segment == 7:
                    ax.set_xticks(range(segment_acceptable_area.shape[1]))
                    ax.set_xticklabels(label_foot, fontdict=FONT_DICT)
                else:
                    ax.get_xaxis().set_visible(False)
        plt.text(-9, -42, 'Z error (mm)', fontdict=FONT_DICT, rotation=90)
        plt.text(-9, -8, 'Y error (mm)', fontdict=FONT_DICT, rotation=90)
        plt.text(-2.5, 16, 'X error (mm)', fontdict=FONT_DICT)
        plt.text(17, -26, 'Z error (mm)', fontdict=FONT_DICT, rotation=90)
        plt.text(18, 16, 'rotation error (degree)', fontdict=FONT_DICT)
        legend_name = ['unacceptable', 'acceptable']
        plt.legend([mpatches.Patch(color=cmap_item(b)) for b in [0, 255]],
                   ['{}'.format(legend_name[i]) for i in range(2)], loc=(-0.2, 5.9), mode={'expand'},
                   fontsize=FONT_SIZE, ncol=2, framealpha=False)
        plt.tight_layout(w_pad=1.05, h_pad=1.05)
        fig.subplots_adjust(left=0.1, right=0.96, bottom=0.1, top=0.88, wspace=0.3)
        file_path = RESULT_PATH + 'result_segment_translation\\' + date + '\\acceptable_area.png'
        plt.savefig(file_path)
        with open(RESULT_PATH + 'result_segment\\' + date + '\\segment_area_all.txt', 'wb') as fp:
            pickle.dump(area_all, fp)

        plt.show()

    @staticmethod
    def show_segment_rotation_result(date):
        with open(RESULT_PATH + 'result_segment_rotation\\' + date + '\\segment_result_all.txt', 'rb') as fp:  # Pickling
            result_all = pickle.load(fp)

        # get min and max of each three figures
        max_value, min_value = np.zeros([3]), np.zeros([3])
        for i_target in range(3):
            # get the max and min of all each target
            target_result = result_all[i_target]
            max_value_target, min_value_target = target_result[0][0, 0], target_result[0][0, 0]
            for im in target_result:
                max_current = np.max(im)
                min_current = np.min(im)
                if max_current > max_value_target:
                    max_value_target = max_current
                if min_current < min_value_target:
                    min_value_target = min_current
            max_value[i_target], min_value[i_target] = max_value_target, min_value_target

        fig, ax = plt.subplots(figsize=(14, 9))
        for i_target in range(3):
            target_result = result_all[i_target]
            for i_result in range(3):
                result_im = target_result[i_result]
                ax = plt.subplot(3, 3, 3*i_target+i_result+1)
                if i_target == 0:
                    plt.imshow(result_im, cmap=plt.cm.get_cmap('RdYlBu'), vmin=min_value[i_target], vmax=max_value[i_target])
                else:
                    plt.imshow(result_im, cmap=plt.cm.get_cmap('RdYlBu_r'), vmin=min_value[i_target], vmax=max_value[i_target])

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i_target == 2:
                    ax.set_xticks(range(result_im.shape[1]))
                    x_tick_label = ['-25°', '', '', '', '', '0°', '', '', '', '', '25°']
                    ax.set_xticklabels(x_tick_label, fontdict=FONT_DICT)
                    ax.get_xaxis().set_visible(True)
                if i_result == 0:
                    ax.set_yticks(range(result_im.shape[0]))
                    ax.set_yticklabels(SEGMENT_NAMES, fontdict=FONT_DICT)
                    ax.get_yaxis().set_visible(True)

        # draw the reference bar
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.75, wspace=0.2, hspace=0.1)
        colorbar_pos = np.array([[0.78, 0.67, 0.02, 0.2],
                                 [0.78, 0.4, 0.02, 0.2],
                                 [0.78, 0.13, 0.02, 0.2]])
        for i_target in range(3):
            ax1 = fig.add_axes(colorbar_pos[i_target, :])
            norm = mpl.colors.Normalize(vmin=min_value[i_target], vmax=max_value[i_target])
            if i_target == 0:
                cb = mpl.colorbar.ColorbarBase(ax1, norm=norm, orientation='vertical', cmap=plt.cm.get_cmap('RdYlBu'))
            else:
                cb = mpl.colorbar.ColorbarBase(ax1, norm=norm, orientation='vertical', cmap=plt.cm.get_cmap('RdYlBu_r'))
            cb.ax.tick_params(labelsize=FONT_SIZE)

        plt.text(-30, 3.8, 'xGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(-19, 3.8, 'yGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(-9, 3.8, 'zGRF', fontdict={'fontsize': FONT_SIZE})
        plt.text(6, 3.6, 'R2 decrease', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(6, 2.3, 'RMSE increase', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        plt.text(6, 0.9, 'NRMSE increase', fontdict={'fontsize': FONT_SIZE}, rotation=90)
        file_path = RESULT_PATH + 'result_segment_rotation\\' + date + '\\segment_rotation.png'
        plt.savefig(file_path)
        plt.show()

    @staticmethod
    def __get_axis_name(segment_name):
        if segment_name in ['trunk', 'pelvis']:
            return [segment_name + '_x_offset', segment_name + '_z_offset']
        elif segment_name in ['l_foot', 'r_foot']:
            return [segment_name + '_x_offset', segment_name + '_y_offset']
        else:
            return [segment_name + '_theta_offset', segment_name + '_z_offset']

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
    def save_segment_translation_result(result_df, date):
        result_all = []
        for target_name in ['R2', 'RMSE', 'NRMSE']:
            result_names = [force_name + '_' + target_name for force_name in ['ForX', 'ForY', 'ForZ']]
            im_all = []
            for segment in SEGMENT_NAMES:
                segment_df = result_df[result_df['segment'] == segment]
                im_all.append(Presenter.__get_segment_result_translation(segment_df, result_names))
            result_all.append(im_all)
        with open(RESULT_PATH + 'result_segment_translation\\' + date + '\\segment_result_all.txt', 'wb') as fp:
            pickle.dump(result_all, fp)

    @staticmethod
    def save_segment_rotation_result(result_df, date):
        result_all = []
        for target_name in ['R2', 'RMSE', 'NRMSE']:
            result_names = [force_name + '_' + target_name for force_name in ['ForX', 'ForY', 'ForZ']]

            result_im = Presenter.__get_segment_result_rotation(result_df, result_names)
            result_all.append(result_im)
        with open(RESULT_PATH + 'result_segment_rotation\\' + date + '\\segment_result_all.txt', 'wb') as fp:
            pickle.dump(result_all, fp)

    @staticmethod
    def show_all_combined_result(result_df, date, folder_name, sub_num=SUB_NUM):
        i_plot = 1
        plt.figure(figsize=(13, 10))
        for target_name in ['R2', 'RMSE', 'NRMSE']:
            result_names = [force_name + '_' + target_name for force_name in ['ForX', 'ForY', 'ForZ']]
            for result_name in result_names:
                result_im = np.zeros([16, 16])
                im_num = len(SPEEDS) * sub_num
                for i_sub in range(sub_num):
                    for speed in SPEEDS:
                        speed_df = result_df[(result_df['subject_id'] == i_sub) & (result_df['speed'] == float(speed))]
                        original_value = np.mean(speed_df[['FP1.' + result_name, 'FP2.' + result_name]].as_matrix()[0, :])
                        result_value = speed_df[['FP1.' + result_name, 'FP2.' + result_name]].as_matrix()
                        result_value = np.mean(result_value[1:, :], axis=1)
                        speed_im = Presenter.__get_all_combined_im(result_value, original_value)
                        result_im = result_im + speed_im
                result_im /= im_num
                plt.subplot(3, 3, i_plot)
                if target_name == 'R2':
                    plt.imshow(result_im, cmap=plt.cm.get_cmap('RdYlBu'))
                else:
                    plt.imshow(result_im, cmap=plt.cm.get_cmap('RdYlBu_r'))
                plt.title(result_name)
                plt.colorbar()
                i_plot += 1
        file_path = RESULT_PATH + folder_name + '\\' + date + '\\result.png'
        plt.savefig(file_path)
        plt.show()

    @staticmethod
    def __get_segment_result_translation(segment_df, result_names, sub_num=SUB_NUM):
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
    def __get_segment_result_rotation(result_df, result_names, sub_num=SUB_NUM):
        axis_value = result_df['trunk_rotation']
        axis_range = list(set(axis_value))
        axis_range.sort()
        axis_len = axis_range.__len__()

        result_im_all = []
        total_number = sub_num * len(SPEEDS)

        result_names_combined = [['FP1.' + name, 'FP2.' + name] for name in result_names]
        for force in result_names_combined:
            added_array = np.zeros([1, axis_len])
            result_im = np.zeros([len(SEGMENT_NAMES), axis_len])
            for i_segment in range(len(SEGMENT_NAMES)):
                segment_df = result_df[result_df['segment'] == SEGMENT_NAMES[i_segment]]
                segment_name = segment_df['segment'].iloc[0]
                axis_name = segment_name + '_rotation'
                for i_sub in range(sub_num):
                    for speed in SPEEDS:
                        trial_df = segment_df[(segment_df['subject_id'] == i_sub) & (segment_df['speed'] == float(speed))]
                        sub_array = Presenter.__get_decrease_array_uni(trial_df[[axis_name, force[0], force[1]]], axis_range, axis_name)
                        added_array += sub_array
                average_array = added_array / total_number
                result_im[i_segment, :] = average_array
            result_im_all.append(result_im)
        return result_im_all

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
    def __get_decrease_array_uni(scores_df, axis_0_range, axis_0_name):
        axis_0_len = axis_0_range.__len__()
        score_im = np.zeros([1, axis_0_len])
        center_df = scores_df[scores_df[axis_0_name] == 0]
        center_score = np.mean(center_df.as_matrix()[0][1:])
        i_score = 0
        for i_y in range(axis_0_len):
            # for image, row and column are exchanged compared to ndarray
            score = scores_df[scores_df[axis_0_name] == axis_0_range[i_y]].as_matrix()
            score_im[0, i_y] = (score[0][-1] + score[0][-2]) / 2 - center_score
            i_score += 1
        return score_im

    @staticmethod
    def __get_all_combined_im(scores, original_value, im_len=16):
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
        ax.set_xticklabels(x_label, fontdict=FONT_DICT, rotation=45)
        ax.set_xlabel(axis_0_label, fontdict=FONT_DICT)
        if axis_1_label.__contains__('theta'):
            y_label = [str(int(tick)) + ' °' for tick in axis_1_range]
        else:
            y_label = [str(int(1000 * tick)) + ' mm' for tick in axis_1_range]
        ax.set_yticks(range(score_im.shape[0]))
        ax.set_yticklabels(y_label, fontdict=FONT_DICT)
        ax.set_ylabel(axis_1_label, fontdict=FONT_DICT)
        # plt.title(title, fontdict=FONT_DICT)
        plt.tight_layout()
        file_path = RESULT_PATH + folder + '\\' + title + '.png'
        plt.savefig(file_path)

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
