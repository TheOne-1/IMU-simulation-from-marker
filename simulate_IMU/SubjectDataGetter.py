# this class is used to get the subject data

import pandas as pd


class SubjectDataGetter:

    def __init__(self, processed_data_path, subject_id):
        self.__subject_id = subject_id
        self.__processed_data_path = processed_data_path
        # speeds = ['0.8', '1.2', '1.6']
        # for speed in speeds:
        #     data[]
        self.__cali_data = {}       # a dict for 3 speeds' cali data
        self.__walking_1_data = {}
        self.__walking_2_data = {}

    def get_all_data(self, speed):
        return self.get_cali_data(speed), self.get_walking_1_data(speed), self.get_walking_2_data(speed)

    def get_cali_data(self, speed):
        cali_file_path = self.__processed_data_path + 'subject_' + str(self.__subject_id) + '\\' + speed + '_cali.csv'
        cali_data_df = pd.read_csv(cali_file_path)
        self.__cali_data[speed] = cali_data_df
        return cali_data_df

    def get_walking_1_data(self, speed):
        walking_file_path_1 = self.__processed_data_path + 'subject_' + \
            str(self.__subject_id) + '\\' + speed + '_walking_1.csv'
        walking_data_1_df = pd.read_csv(walking_file_path_1)
        return walking_data_1_df

    def get_walking_2_data(self, speed):
        walking_file_path_2 = self.__processed_data_path + 'subject_' + \
            str(self.__subject_id) + '\\' + speed + '_walking_2.csv'
        walking_data_2_df = pd.read_csv(walking_file_path_2)
        return walking_data_2_df
