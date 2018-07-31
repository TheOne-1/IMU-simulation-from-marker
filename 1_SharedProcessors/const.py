FORCE_PLATE_THRESHOLD = 50      # set 50N as the threshold of the force plate

DO_SCALING = True
DO_PCA = False
DO_SHUFFLING = True

NORMALIZE_ACC = True
NORMALIZE_GRF = True

RAW_DATA_PATH = 'D:\Tian\Research\Projects\ML Project\gait_database\GaitDatabase\data\\'
PROCESSED_DATA_PATH = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data\\'
RESULT_PATH = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\\result\\'
ALL_SUB_FILE = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data_all_sub\\all_sub.csv'

N_COMPONENT = 12
SUB_NUM = 10
SPEEDS = ['0.8', '1.2', '1.6']
SPEED_NUM = SPEEDS.__len__()
SEGMENT_NAMES = ['trunk', 'pelvis', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank', 'l_foot', 'r_foot']
SENSOR_NUM = SEGMENT_NAMES.__len__()
MOCAP_SAMPLE_RATE = 100
THIGH_COEFF = 1.1  # D_thigh = THIGH_COEFF * knee_width
SHANK_COEFF = 1  # D_shank = SHANK_COEFF * D_great_trochanter

WN_MARKER = 10 / (MOCAP_SAMPLE_RATE / 2)
WN_FORCE = 25 / (MOCAP_SAMPLE_RATE / 2)

# IM_VMIN = 0.5
# IM_VMAX = 1
FONT_SIZE = 12

ALL_ACC_NAMES = ['trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
                 'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
                 'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
                 'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
                 'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
                 'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
                 'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
                 'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z']

ALL_ACC_GYR_NAMES = ['trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
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
                     'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z']

ALL_FORCE_NAMES = ['FP1.ForX', 'FP2.ForX', 'FP1.ForY', 'FP2.ForY', 'FP1.ForZ', 'FP2.ForZ']

OFFSET_COLUMN_NAMES = ['trunk_x_offset', 'trunk_y_offset', 'trunk_z_offset', 'trunk_theta_offset',
                       'pelvis_x_offset', 'pelvis_y_offset', 'pelvis_z_offset', 'pelvis_theta_offset',
                       'l_thigh_x_offset', 'l_thigh_y_offset', 'l_thigh_z_offset', 'l_thigh_theta_offset',
                       'r_thigh_x_offset', 'r_thigh_y_offset', 'r_thigh_z_offset', 'r_thigh_theta_offset',
                       'l_shank_x_offset', 'l_shank_y_offset', 'l_shank_z_offset', 'l_shank_theta_offset',
                       'r_shank_x_offset', 'r_shank_y_offset', 'r_shank_z_offset', 'r_shank_theta_offset',
                       'l_foot_x_offset', 'l_foot_y_offset', 'l_foot_z_offset', 'l_foot_theta_offset',
                       'r_foot_x_offset', 'r_foot_y_offset', 'r_foot_z_offset', 'r_foot_theta_offset']





















