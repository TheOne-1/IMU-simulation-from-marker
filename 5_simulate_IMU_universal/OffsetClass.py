import numpy as np
import pandas as pd
from VirtualProcessor import Processor
from const import *


# very basic offset class
class Offset:
    def __init__(self, segment, x_offset=0, y_offset=0, z_offset=0, theta_offset=0, R=None):
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

    @staticmethod
    def combine_segment_offset(offset_0, offset_1):
        segment_0 = offset_0.get_segment()
        segment_1 = offset_0.get_segment()
        if segment_0 != segment_1:
            raise RuntimeError('two offset segment must be the same')
        transform_0 = offset_0.get_all_offsets()
        transform_1 = offset_1.get_all_offsets()
        transform = transform_0 + transform_1

        if offset_0.get_rotation() is not None:
            R = offset_0.get_rotation()
        else:
            R = offset_1.get_rotation()
        offset = Offset(segment_0, transform[0], transform[1], transform[2], transform[3], R)
        return offset


class OneAxisOffset:
    def __init__(self, segment, axis_name, iterable_object, diameter=None):
        self.__segment = segment
        self.__i_offset = 0

        if axis_name not in ['x', 'y', 'z', 'theta']:
            raise RuntimeError('Wrong axis name.')
        offsets = []
        if axis_name is 'x':
            for item in iterable_object:
                offset = Offset(self.__segment, x_offset=item / 1000)       # change millimeter to meter
                offsets.append(offset)
        if axis_name is 'y':
            for item in iterable_object:
                offset = Offset(self.__segment, y_offset=item / 1000)       # change millimeter to meter
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

    @staticmethod
    def get_segment_multi_axis_offset(segment, thigh_diameter, shank_diameter):
        multi_offset = MultiAxisOffset()
        if segment in ['trunk', 'pelvis']:
            x_range = range(-100, 101, 20)
            x_axis = OneAxisOffset(segment, 'x', x_range)
            multi_offset.add_offset_axis(x_axis)
            z_range = range(-100, 101, 20)
            z_axis = OneAxisOffset(segment, 'z', z_range)
            multi_offset.add_offset_axis(z_axis)
        elif segment in ['l_thigh', 'r_thigh']:
            theta_range = range(-25, 26, 5)
            theta_axis = OneAxisOffset(segment, 'theta', theta_range, diameter=thigh_diameter)
            multi_offset.add_offset_axis(theta_axis)
            z_range = range(-100, 101, 20)
            z_axis = OneAxisOffset(segment, 'z', z_range)
            multi_offset.add_offset_axis(z_axis)
        elif segment in ['l_shank', 'r_shank']:
            theta_range = range(-25, 26, 5)
            theta_axis = OneAxisOffset(segment, 'theta', theta_range, diameter=shank_diameter)
            multi_offset.add_offset_axis(theta_axis)
            z_range = range(-100, 101, 20)
            z_axis = OneAxisOffset(segment, 'z', z_range)
            multi_offset.add_offset_axis(z_axis)
        else:
            x_range = range(-50, 51, 10)
            x_axis = OneAxisOffset(segment, 'x', x_range)
            multi_offset.add_offset_axis(x_axis)
            y_range = range(-50, 51, 10)
            y_axis = OneAxisOffset(segment, 'y', y_range)
            multi_offset.add_offset_axis(y_axis)
        return multi_offset


# this class is used to return offset combinations between all 8 segments
class SegmentCombos:
    def __init__(self, error_combo, thigh_diameter, shank_diameter):
        if error_combo.shape != (8, 2):
            raise RuntimeError('Incorrect offset length')

        offset_0 = SegmentCombos.__initialize_both_axes_offset('trunk', 'x', error_combo[0, 0], 'z', error_combo[0, 1])
        offset_1 = SegmentCombos.__initialize_both_axes_offset('trunk', 'x', -error_combo[0, 0], 'z', -error_combo[0, 1])
        trunk_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('pelvis', 'x', error_combo[1, 0], 'z', error_combo[1, 1])
        offset_1 = SegmentCombos.__initialize_both_axes_offset('pelvis', 'x', -error_combo[1, 0], 'z', -error_combo[1, 1])
        pelvis_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('l_thigh', 'theta', error_combo[2, 0], 'z', error_combo[2, 1], diameter=thigh_diameter)
        offset_1 = SegmentCombos.__initialize_both_axes_offset('l_thigh', 'theta', -error_combo[2, 0], 'z', -error_combo[2, 1], diameter=thigh_diameter)
        l_thigh_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('r_thigh', 'theta', error_combo[3, 0], 'z', error_combo[3, 1], diameter=thigh_diameter)
        offset_1 = SegmentCombos.__initialize_both_axes_offset('r_thigh', 'theta', -error_combo[3, 0], 'z', -error_combo[3, 1], diameter=thigh_diameter)
        r_thigh_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('l_shank', 'theta', error_combo[4, 0], 'z', error_combo[4, 1], diameter=shank_diameter)
        offset_1 = SegmentCombos.__initialize_both_axes_offset('l_shank', 'theta', -error_combo[4, 0], 'z', -error_combo[4, 1], diameter=shank_diameter)
        l_shank_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('r_shank', 'theta', error_combo[5, 0], 'z', error_combo[5, 1], diameter=shank_diameter)
        offset_1 = SegmentCombos.__initialize_both_axes_offset('r_shank', 'theta', -error_combo[5, 0], 'z', -error_combo[5, 1], diameter=shank_diameter)
        r_shank_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('l_foot', 'x', error_combo[6, 0], 'y', error_combo[6, 1])
        offset_1 = SegmentCombos.__initialize_both_axes_offset('l_foot', 'x', -error_combo[6, 0], 'y', -error_combo[6, 1])
        l_foot_list = [offset_0, offset_1]

        offset_0 = SegmentCombos.__initialize_both_axes_offset('r_foot', 'x', error_combo[7, 0], 'y', error_combo[7, 1])
        offset_1 = SegmentCombos.__initialize_both_axes_offset('r_foot', 'x', -error_combo[7, 0], 'y', -error_combo[7, 1])
        r_foot_list = [offset_0, offset_1]

        self.__list_all = [trunk_list, pelvis_list, l_thigh_list, r_thigh_list, l_shank_list, r_shank_list,
                           l_foot_list, r_foot_list]

    def get_offset_df(self):
        offset_combos = self.get_segment_combos()
        zero_row = np.zeros([1, len(OFFSET_COLUMN_NAMES)])
        # add a zero line to store no offset result
        offset_df = pd.DataFrame(zero_row.copy(), columns=OFFSET_COLUMN_NAMES)
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

    # this function returns combos between different segments
    def get_segment_combos(self):
        x = SegmentCombos.__get_segment_combos_recursive(self.__list_all)
        return x


    @staticmethod
    def __get_segment_combos_recursive(list_all):
        if len(list_all) == 1:
            return_list = [[list_all[0][0]], [list_all[0][1]]]
        else:
            return_list = []
            first_segment = list_all[0]
            last_return_list = SegmentCombos.__get_segment_combos_recursive(list_all[1:])
            for last_combo in last_return_list:
                new_combo = [first_segment[0]]
                new_combo.extend(last_combo)
                return_list.append(new_combo)
                new_combo = [first_segment[1]]
                new_combo.extend(last_combo)
                return_list.append(new_combo)
        return return_list

    @staticmethod
    def __initialize_offset(segment, axis_name, item, diameter=None):
        if axis_name not in ['x', 'y', 'z', 'theta']:
            raise RuntimeError('Wrong axis name.')

        if axis_name is 'x':
            offset = Offset(segment, x_offset=item / 1000)  # change millimeter to meter
        elif axis_name is 'y':
            offset = Offset(segment, y_offset=item / 1000)  # change millimeter to meter
        elif axis_name is 'z':
            offset = Offset(segment, z_offset=item / 1000)
        else:
            if not diameter:
                raise RuntimeError('Missing the cylinder diameter.')
            theta_radians = item * np.pi / 180  # change degree to radians
            R_cylinder = Processor.get_cylinder_surface_rotation(theta_radians)
            x = diameter / 2 * (1 - np.cos(theta_radians))
            y = diameter / 2 * np.sin(theta_radians)
            offset = Offset(segment=segment, x_offset=x, y_offset=y, theta_offset=item, R=R_cylinder)
        return offset


    @staticmethod
    def __initialize_both_axes_offset(segment, axis_0_name, item_0, axis_1_name, item_1, diameter=None):
        offset_0 = SegmentCombos.__initialize_offset(segment, axis_0_name, item_0, diameter)
        offset_1 = SegmentCombos.__initialize_offset(segment, axis_1_name, item_1, diameter)

        offset_value_0 = offset_0.get_all_offsets()
        offset_value_1 = offset_1.get_all_offsets()
        offset_value = offset_value_0 + offset_value_1

        if offset_0.get_rotation() is not None:
            R = offset_0.get_rotation()
        elif offset_1.get_rotation() is not None:
            R = offset_1.get_rotation()
        else:
            R = None

        offset = Offset(segment, offset_value[0], offset_value[1], offset_value[2], offset_value[3], R)
        return offset




















