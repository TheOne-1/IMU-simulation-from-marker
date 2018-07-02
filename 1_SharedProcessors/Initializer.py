from numpy import *
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
from const import *


class Initializer:
    # to solve B = R * A + t, R is Rba
    @staticmethod
    def rigid_transform_3D(A, B):
        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = mean(A, axis=0)
        centroid_B = mean(B, axis=0)
        # centre the points
        AA = A - tile(centroid_A, (N, 1))
        BB = B - tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = dot(AA.T, BB)
        U, S, Vt = linalg.svd(H)
        R = dot(Vt.T, U.T)

        # special reflection case
        if linalg.det(R) < 0:
            # print
            # "Reflection detected"
            Vt[2, :] *= -1
            R = dot(Vt.T, U.T)
        t = -dot(R, centroid_A.T) + centroid_B.T
        return R, t

    @staticmethod
    def filtering(data, wn, filter_order=4):
        b, a = butter(filter_order, wn, btype='low')
        return filtfilt(b, a, data, axis=0)

    # to change the database coordinate into our lab's coordinate
    @staticmethod
    def __coordinate_transfer(data):
        # turn z around
        data[:, 2] = -data[:, 2]
        # switch y and z
        temp = np.array(data[:, 2])
        data[:, 2] = data[:, 1]
        data[:, 1] = temp
        return data

    @staticmethod
    def process_data(data_df, marker_column_num, force_column_num, filter_order=4):
        column_names = data_df.columns
        data = data_df.as_matrix()

        # marker filtering
        marker_unfilt = data[:, marker_column_num]
        marker_num = int(marker_unfilt.shape[1] / 3)
        for i_marker in range(marker_num):
            marker_unfilt[:, 3*i_marker:3*(i_marker+1)] = \
                Initializer.__coordinate_transfer(marker_unfilt[:, 3*i_marker:3*(i_marker+1)])
        marker_filt = Initializer.filtering(marker_unfilt, WN_MARKER, filter_order)
        data[:, marker_column_num] = marker_filt

        # do filtering to the force first, then set zero, then set filter the COP, otherwise incorrect cop
        # in swing phase will affect the COP in stance phase
        # coordinate transfer
        force_unfilt = data[:, force_column_num]
        for i_force_plate in range(4):      # there are two forces, two cops
            force_unfilt[:, 3*i_force_plate:3*(i_force_plate+1)] = \
                Initializer.__coordinate_transfer(force_unfilt[:, 3*i_force_plate:3*(i_force_plate+1)])
        force_filt = force_unfilt.copy()
        # force filtering
        force_filt[:, 3:6] = Initializer.filtering(force_unfilt[:, 3:6], WN_FORCE, filter_order)
        force_filt[:, 9:12] = Initializer.filtering(force_unfilt[:, 9:12], WN_FORCE, filter_order)
        # set swing phase to zero
        force_1_norm = np.linalg.norm(force_filt[:, 3:6], axis=1)
        plate_1_zero_index = np.where(force_1_norm < FORCE_PLATE_THRESHOLD)
        force_2_norm = np.linalg.norm(force_filt[:, 9:12], axis=1)
        plate_2_zero_index = np.where(force_2_norm < FORCE_PLATE_THRESHOLD)
        force_filt[plate_1_zero_index, 0:6] = 0       # zero data after filtering
        force_filt[plate_2_zero_index, 6:12] = 0
        # cop filtering
        force_filt[:, 0:3] = Initializer.filtering(force_filt[:, 0:3], WN_FORCE, filter_order)
        force_filt[:, 6:9] = Initializer.filtering(force_filt[:, 6:9], WN_FORCE, filter_order)
        force_filt[plate_1_zero_index, 0:3] = 0       # zero cop data again after filtering
        force_filt[plate_2_zero_index, 9:12] = 0
        data[:, force_column_num] = force_filt

        data_df = pd.DataFrame(data)
        data_df.columns = column_names
        return data_df
