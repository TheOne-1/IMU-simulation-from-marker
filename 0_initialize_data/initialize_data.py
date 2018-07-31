# this project is designed to process the opensource gait database
from gaitanalysis.motek import DFlowData
from const import *
from DatabaseInfo import *
from Initializer import Initializer
from SaveData import SaveData

speed_num = SPEEDS.__len__()

data_path = 'D:\Tian\Research\Projects\ML Project\gait_database\GaitDatabase\data\\'
processed_data_path = 'D:\Tian\Research\Projects\ML Project\gait_database_processed\GaitDatabase\data\\'
my_database_info = DatabaseInfo()
my_initializer = Initializer()
necessary_columns = my_database_info.get_necessary_columns()
marker_column_num = my_database_info.get_marker_column_num()        # get column numbers
force_column_num = my_database_info.get_force_column_num()
all_column_names = my_database_info.get_all_column_names()

for i_sub in range(SUB_NUM):
    print('sub: ' + str(i_sub))
    for i_speed in range(speed_num):
        file_names = my_database_info.get_file_names(sub=i_sub, speed=i_speed, path=data_path)
        trial_data = DFlowData(file_names[0], file_names[1], file_names[2])
        event_dictionary = trial_data.meta['trial']['events']

        trial_data.clean_data(interpolation_order=2)
        # events: A: Force Plate Zeroing    B: Calibration Pose     C: First Normal Walking
        # D: Longitudinal Perturbation      E: Second Normal Walking    F: Unloaded End

        cali_data_df = trial_data.extract_processed_data(event='Calibration Pose')[necessary_columns]
        walking_data_df_1 = trial_data.extract_processed_data(event='First Normal Walking')[necessary_columns]
        walking_data_df_2 = trial_data.extract_processed_data(event='Second Normal Walking')[necessary_columns]

        # cali_data_df.columns = all_column_names
        # walking_data_df_1.columns = all_column_names

        cali_processed = my_initializer.process_data(cali_data_df, marker_column_num, force_column_num)
        walking_1_processed = my_initializer.process_data(walking_data_df_1, marker_column_num, force_column_num)
        walking_2_processed = my_initializer.process_data(walking_data_df_2, marker_column_num, force_column_num)

        my_subject_data = SaveData(SPEEDS[i_speed])
        save_path = processed_data_path + 'subject_' + str(i_sub) + '\\'
        my_subject_data.save_data(cali_processed, walking_1_processed, walking_2_processed, save_path)


































