import pickle

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import xlwt

from DatabaseInfo import DatabaseInfo
from Presenter import Presenter
from OffsetClass import *
from const import *


class PresenterTransRota:
    @staticmethod
    def save_segment_trans_rota(result_df, date):
        result_all_targets = {}
        for target_name in ['R2', 'RMSE', 'NRMSE']:
            result_names = [force_name + '_' + target_name for force_name in ['ForX', 'ForY', 'ForZ']]
            result_all_segments = {}
            for segment in SEGMENT_NAMES:
                segment_df = result_df[result_df['segment'] == segment]
                result_all_segments[segment] = PresenterTransRota.__get_segment_result_trans_rota(segment_df, result_names)
            result_all_targets[target_name] = result_all_segments
        with open(RESULT_PATH + 'result_segment_trans_rota\\' + date + '\\segment_result_all.txt', 'wb') as fp:
            pickle.dump(result_all_targets, fp)

    @staticmethod
    def show_segment_trans_rota(result_name, date, model_name='Gradient Boosting'):
        with open(RESULT_PATH + 'result_segment_trans_rota\\' + date + '\\segment_result_all.txt', 'rb') as fp:
            result_all = pickle.load(fp)

        force_title = ['MLGRF', 'APGRF', 'VGRF']
        mm_label = ['-100mm', '', '0mm', '', '100mm']
        theta_label = ['-25°', '', '0°', '', '25°']

        result_all_segments = result_all[result_name]
        for i_segment in range(8):
            plt.subplots(figsize=(9, 8))
            for i_axis in range(3):
                for i_force in range(3):
                    ax = plt.subplot(3, 3, 3*i_axis+i_force+1)
                    axes_names = PresenterTransRota.__get_axis_names(SEGMENT_NAMES[i_segment])
                    im = result_all_segments[SEGMENT_NAMES[i_segment]][axes_names[i_axis]][i_force]
                    plt.imshow(im)
                    if i_axis == 0:
                        plt.title(force_title[i_force])
                    axes_names.pop(i_axis)
                    ax.set_xlabel(axes_names[0])
                    ax.set_ylabel(axes_names[1])
                    ax.set_xticks(range(5))
                    ax.set_yticks(range(5))
                    if axes_names[0].__contains__('theta') or axes_names[0].__contains__('rotation'):
                        ax.set_xticklabels(theta_label)
                    else:
                        ax.set_xticklabels(mm_label)
                    if axes_names[1].__contains__('theta') or axes_names[1].__contains__('rotation'):
                        ax.set_yticklabels(theta_label)
                    else:
                        ax.set_yticklabels(mm_label)
            plt.suptitle(SEGMENT_NAMES[i_segment])
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    @staticmethod
    def __get_segment_result_trans_rota(segment_df, result_names, sub_num=SUB_NUM):
        segment_name = segment_df['segment'].iloc[0]
        axes_names = PresenterTransRota.__get_axis_names(segment_name)
        result_all_three_axes = {}
        for i_axis in range(3):
            other_axis_names = axes_names.copy()
            other_axis_names.pop(i_axis)
            axis_name = axes_names[i_axis]
            segment_df_axis = segment_df[segment_df[axis_name] == 0]
            axis_result = Presenter._get_segment_result_translation(segment_df_axis, result_names, axes_names=other_axis_names)
            result_all_three_axes[axis_name] = axis_result
        return result_all_three_axes


    @staticmethod
    def __get_axis_names(segment_name):
        axis_names = Presenter._get_axis_name(segment_name)
        axis_names.append(segment_name + '_rotation')
        return axis_names










