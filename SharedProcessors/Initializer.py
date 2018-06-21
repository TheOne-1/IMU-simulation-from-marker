from numpy import *
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np


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

    def filtering(self, data, wn, filter_order=4):
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

    def process_data(self, data_df, marker_column_num, force_column_num, wn_marker, wn_force, filter_order=4):
        column_names = data_df.columns
        data = data_df.as_matrix()

        # marker filtering
        marker_unfilt = data[:, marker_column_num] * 1000       # transfer to meter
        marker_num = int(marker_unfilt.shape[1] / 3)
        for i_marker in range(marker_num):
            marker_unfilt[:, 3*i_marker:3*(i_marker+1)] = self.__coordinate_transfer(marker_unfilt[:, 3*i_marker:3*(i_marker+1)])
        marker_filt = self.filtering(marker_unfilt, wn_marker, filter_order)
        data[:, marker_column_num] = marker_filt

        # force filtering
        force_unfilt = data[:, force_column_num]
        force_filt = self.filtering(force_unfilt, wn_force, filter_order)
        data[:, force_column_num] = force_filt

        data_df = pd.DataFrame(data)
        data_df.columns = column_names
        return data_df