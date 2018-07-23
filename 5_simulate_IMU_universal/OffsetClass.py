import numpy as np
import pandas as pd
from VirtualProcessor import Processor
from const import *


# very basic offset class
class Offset:
    def __init__(self, segment, x_offset=0, z_offset=0, y_offset=0, theta_offset=0, R=None):
        self.__segment = segment
        self.__x_offset = x_offset
        self.__z_offset = z_offset
        self.__y_offset = y_offset
        self.__theta_offset = theta_offset
        self.__R = R

    def get_segment(self):
        return self.__segment

    def get_offset_names(self):
        suffix = ['_x_offset', '_y_offset', '_z_offset', '_theta_offset']
        names = [self.__segment + item for item in suffix]
        return names

    def get_rotation(self):
        return self.__R

    def get_translation(self):
        return np.array([self.__x_offset, self.__y_offset, self.__z_offset])

    def get_all_offsets(self):
        return np.array([self.__x_offset, self.__y_offset, self.__z_offset, self.__theta_offset])

    def to_string(self):
        if self.__R is not None:
            str_R = 'True'
        else:
            str_R = 'False'
            print('segment: ' + str(self.__segment) + ';\tx offset: ' + str(self.__x_offset) + ';\ty offset: ' +
                  str(self.__y_offset) + ';\tz offset: ' + str(self.__z_offset) + ';\tR: ' + str_R)


class OneAxisOffset:
    def __init__(self, segment, axis_name, iterable_object, diameter=None):
        self.__segment = segment
        self.__i_offset = 0

        if axis_name not in ['x', 'z', 'theta']:
            raise RuntimeError('Wrong axis name.')
        offsets = []
        if axis_name is 'x':
            for item in iterable_object:
                offset = Offset(self.__segment, x_offset=item / 1000)       # change millimeter to meter
                offsets.append(offset)
        if axis_name is 'z':
            for item in iterable_object:
                offset = Offset(self.__segment, z_offset=item / 1000)
                offsets.append(offset)
        if axis_name is 'theta':

            if not diameter:
                raise RuntimeError('Missing the cylinder diameter.')
            for item in iterable_object:
                theta_radians = item * np.pi / 180  # change degree to radians
                R_cylinder = Processor.get_cylinder_surface_rotation(theta_radians)
                x = diameter / 2 * (1 - np.cos(theta_radians))
                y = diameter / 2 * np.sin(theta_radians)
                offset = Offset(segment=segment, x_offset=x, y_offset=y, theta_offset=item, R=R_cylinder)
                offsets.append(offset)
        self.__offsets = offsets

    def __iter__(self):
        return self

    def __next__(self):
        if self.__i_offset == len(self.__offsets):
            self.__i_offset = 0
            raise StopIteration
        self.__i_offset += 1
        return self.__offsets[self.__i_offset - 1]


class MultiAxisOffset:
    def __init__(self):
        self.__axes = []
        self.__axis_num = 0

    def add_offset_axis(self, axis):
        self.__axes.append(axis)
        self.__axis_num += 1

    def get_combos(self):
        offset_list = MultiAxisOffset.__get_combos_recursive(self.__axes)
        return offset_list

    # reorganize the axes by giving every combinations
    @staticmethod
    def __get_combos_recursive(axes):
        if len(axes) == 0:
            return []
        elif len(axes) == 1:
            return_list = []
            for offset in axes[0]:
                return_list.append([offset])      # add a [] to make each item iterable
            return return_list
        else:
            return_list = []
            return_list_last = MultiAxisOffset.__get_combos_recursive(axes[1:])
            for axis_item in axes[0]:
                for last_combine in return_list_last:
                    new_combination = [axis_item]
                    new_combination.extend(last_combine)
                    return_list.append(new_combination)
            return return_list

    def get_offset_df(self):
        offset_combos = self.get_combos()
        offset_df = pd.DataFrame(columns=OFFSET_COLUMN_NAMES)
        zero_row = np.zeros([1, len(OFFSET_COLUMN_NAMES)])
        for combo in offset_combos:
            df_item = pd.DataFrame(zero_row, columns=OFFSET_COLUMN_NAMES)
            for offset in combo:
                offset_names = offset.get_offset_names()
                all_offsets = offset.get_all_offsets()
                df_item.loc[0, offset_names] = all_offsets
            offset_df = offset_df.append(df_item)
        offset_df = offset_df.reset_index(drop=True)
        return offset_df
























