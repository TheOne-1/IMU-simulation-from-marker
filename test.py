# this file is used to evaluate all the thigh marker position and find the best location
from datetime import datetime
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing

from DatabaseInfo import DatabaseInfo
from EvaluationUniClass import EvaluationUni
from OffsetClass import *
import matplotlib.pyplot as plt

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    # 'FP1.CopX', 'FP1.CopY',
    # 'FP2.CopX', 'FP2.CopY'
]

R2_column = [output + '_R2' for output in output_names]
RMSE_column = [output + '_RMSE' for output in output_names]
NRMSE_column = [output + '_NRMSE' for output in output_names]
result_column = R2_column + RMSE_column + NRMSE_column

input_names = [
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
    'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
    'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
    'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
    'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
    'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
    'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
    'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
    'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
    'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
]

model = ensemble.RandomForestRegressor(n_jobs=6)
# model = ensemble.RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=400)
# model = ensemble.GradientBoostingRegressor(
#     learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)
x_scalar, y_scalar = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)

x = all_sub_data['FP1.ForZ'].as_matrix()
plt.plot(x)
plt.show()