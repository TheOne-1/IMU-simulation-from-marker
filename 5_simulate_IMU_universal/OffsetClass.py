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
            df_item = pd.DataFrame(zero_row.copy(), columns=OFFSET_COLUMN_NAMES)
            for offset in combo:
                offset_names = offset.get_offset_names()
                all_offsets = offset.get_all_offsets()
                for i_dim in range(len(offset_names)):
                    if df_item.loc[0, offset_names[i_dim]] == 0:
                        df_item.loc[0, offset_names[i_dim]] = all_offsets[i_dim]
            offset_df = offset_df.append(df_item)
        offset_df = offset_df.reset_index(drop=True)
        return offset_df


# this class is used to return offset combinations between all 8 segments
class SegmentOffsets:

    def __init__(self, offset_value_list):
        if len(offset_value_list) != 16:
            raise RuntimeError('Incorrect offset length')

        # add trunk offset
        trunk_list = []
        trunk_list.append(Offset('trunk', x_offset=offset_value_list[0]))
        trunk_list.append(Offset('trunk', z_offset=offset_value_list[1]))
        self.__trunk_list = trunk_list

        # add pelvis offset
        pelvis_list = []
        pelvis_list.append(Offset('pelvis', x_offset=offset_value_list[2]))
        pelvis_list.append(Offset('pelvis', z_offset=offset_value_list[3]))
        self.__pelvis_list = pelvis_list

        # add l_thigh offset
        l_thigh_list = []
        l_thigh_list.append(Offset('l_thigh', theta_offset=offset_value_list[4]))
        l_thigh_list.append(Offset('l_thigh', z_offset=offset_value_list[5]))
        self.__l_thigh_list = l_thigh_list

        # add r_thigh offset
        r_thigh_list = []
        r_thigh_list.append(Offset('r_thigh', theta_offset=offset_value_list[6]))
        r_thigh_list.append(Offset('r_thigh', z_offset=offset_value_list[7]))
        self.__r_thigh_list = r_thigh_list

        # add l_shank offset
        l_shank_list = []
        l_shank_list.append(Offset('l_shank', theta_offset=offset_value_list[8]))
        l_shank_list.append(Offset('l_shank', z_offset=offset_value_list[9]))
        self.__l_shank_list = l_shank_list

        # add r_shank offset
        r_shank_list = []
        r_shank_list.append(Offset('r_shank', theta_offset=offset_value_list[10]))
        r_shank_list.append(Offset('r_shank', z_offset=offset_value_list[11]))
        self.__r_shank_list = r_shank_list

        self.__list_all = [trunk_list, pelvis_list, l_thigh_list, r_thigh_list, l_shank_list, r_shank_list]



    # this function returns combos between different segments
    def get_segment_combos(self):
        x = SegmentOffsets.__get_segment_combos_recursive(self.__list_all)
        return x


    @staticmethod
    def __get_segment_combos_recursive(list_all):
        if len(list_all) == 1:
            return_list = [list_all[0], list_all[1]]
        else:
            return_list = []
            first_segment = list_all[0]
            last_return_list = SegmentOffsets.__get_segment_combos_recursive(list_all[1:])
            for last_combo in last_return_list:
                new_combo = first_segment[0]
                new_combo.extend(last_combo)
                return_list.append(new_combo)
                new_combo = first_segment[1]
                new_combo.extend(last_combo)
                return_list.append(new_combo)

        return return_list























