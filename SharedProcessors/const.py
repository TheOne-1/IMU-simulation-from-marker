
PROCESSED_DATA_PATH = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data\\'


SUB_NUM = 10
SPEEDS = ['0.8', '1.2', '1.6']
SPEED_NUM = SPEEDS.__len__()
SEGMENT_NAMES = ['trunk', 'pelvis', 'l_thigh', 'r_thigh', 'l_shank', 'r_shank', 'l_feet', 'r_feet']
SENSOR_NUM = SEGMENT_NAMES.__len__()
MOCAP_SAMPLE_RATE = 100
THIGH_COEFF = 0.4       # D_thigh = THIGH_COEFF * D_great_trochanter
SHANK_COEFF = 0.3       # D_shank = SHANK_COEFF * D_great_trochanter