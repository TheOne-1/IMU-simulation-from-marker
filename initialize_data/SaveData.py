# this class is used to save subject data of that specific database

import os


class SaveData:

    def __init__(self, speed):
        self.__speed = speed

    def save_data(self, cali_data, walking_data_1, walking_data_2, path):
        cali_file_path = path + self.__speed + '_cali.csv'
        cali_data.to_csv(cali_file_path)

        walking_file_path_1 = path + self.__speed + '_walking_1.csv'
        walking_data_1.to_csv(walking_file_path_1)

        walking_file_path_2 = path + self.__speed + '_walking_2.csv'
        walking_data_2.to_csv(walking_file_path_2)
