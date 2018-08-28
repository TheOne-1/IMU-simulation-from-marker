# train, test data is not stored in this Class. Only models and scalars are stored
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from const import *
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import os
from sklearn.decomposition import PCA
import json
from sklearn.metrics import mean_squared_error
from math import sqrt


class Evaluation:
    def __init__(self, y_column_names, base_model):
        self.__params_column_names = y_column_names
        self.__base_model = base_model
        self.__batch_size = 50  # the size of data that be trained together
        self.__base_scaler = preprocessing.StandardScaler()
        self.__pca = PCA(n_components=N_COMPONENT)

    def train_sklearn(self, x_train, y_train):
        if DO_SCALING:
            self.__x_scalar = sklearn.clone(self.__base_scaler)
            x_train = self.__x_scalar.fit_transform(x_train)
        if DO_PCA:
            self.__pca.fit(x_train)
            x_train = self.__pca.transform(x_train)
        if DO_SHUFFLING:
            x_train, y_train = shuffle(x_train, y_train)

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.as_matrix()
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.as_matrix()

        self.__models = []
        for i_model in range(y_train.shape[1]):
            model = sklearn.clone(self.__base_model)
            model.fit(x_train, y_train[:, i_model])
            self.__models.append(model)

    def evaluate_sklearn(self, x_test, y_test):
        if DO_SCALING:
            x_test = self.__x_scalar.transform(x_test)
        else:
            x_test = x_test.as_matrix()
        if DO_PCA:
            x_test = self.__pca.transform(x_test)
            
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.as_matrix()
        
        y_pred = y_test.copy()
        i_output = 0
        for model in self.__models:
            y_pred[:, i_output] = model.predict(x_test)
            i_output += 1
        return y_pred

    @staticmethod
    def get_all_scores(y_test, y_pred):
        R2 = Evaluation.get_R2(y_test, y_pred)
        RMSE = Evaluation.get_RMSE(y_test, y_pred)
        NRMSE = Evaluation.get_NRMSE(y_test, y_pred)
        return R2, RMSE, NRMSE

    @staticmethod
    def get_R2(y_test, y_pred):
        output_num = y_test.shape[1]
        R2 = []
        for i_output in range(output_num):
            score = r2_score(y_test[:, i_output], y_pred[:, i_output])
            R2.append(score)
        return R2

    @staticmethod
    def get_RMSE(y_test, y_pred):
        output_num = y_test.shape[1]
        RMSEs = []
        for i_output in range(output_num):
            RMSE = sqrt(mean_squared_error(y_test[:, i_output], y_pred[:, i_output]))
            RMSEs.append(RMSE)
        return RMSEs

    @staticmethod
    def get_NRMSE(y_test, y_pred):
        output_num = y_test.shape[1]
        NRMSEs = []
        for i_output in range(output_num):
            RMSE = sqrt(mean_squared_error(y_test[:, i_output], y_pred[:, i_output]))
            NRMSE = 100 * RMSE / (max(y_test[:, i_output]) - min(y_test[:, i_output]))
            NRMSEs.append(NRMSE)
        return NRMSEs

    @staticmethod
    # save the current result to the total result
    def scores_df_item(scores, total_result_columns, subject_id, X_NORM_ALL, Y_NORM):
        result_item = [subject_id, X_NORM_ALL, Y_NORM]
        result_item.extend(scores)
        df_item = pd.DataFrame([result_item], columns=total_result_columns)
        return df_item


    @staticmethod
    def initialize_result_df(total_result_columns):
        return pd.DataFrame(columns=total_result_columns)

    @staticmethod
    def show_result(score_list, xy_generator, output_names):
        axis_1_range, axis_2_range = xy_generator.get_point_range()
        # change result as an image
        result_im = np.zeros([axis_2_range.__len__(), axis_1_range.__len__()])
        for i_output in range(output_names.__len__()):
            i_y, i_result = 0, 0
            for i_x in range(axis_1_range.__len__()):
                for i_y in range(axis_2_range.__len__()):
                    # for image, row and column are exchanged compared to ndarray
                    scores = score_list[i_result][0]        # the each element in score_list is the data of that point
                    score = scores[i_output]
                    result_im[i_y, i_x] = score
                    i_result += 1
            # plt.figure()
            fig, ax = plt.subplots()
            im = plt.imshow(result_im)
            plt.colorbar(im)
            x_label = list(axis_1_range)
            ax.set_xticks(range(result_im.shape[1]))
            ax.set_xticklabels(x_label)
            y_label = list(axis_2_range)
            ax.set_yticks(range(result_im.shape[0]))
            ax.set_yticklabels(y_label)
            plt.title(output_names[i_output])
        plt.show()
        return result_im

    @staticmethod
    def save_result(total_result_df, folder_name):
        date = time.strftime('%Y%m%d')
        file_path = RESULT_PATH + folder_name + '\\' + date + '.csv'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = RESULT_PATH + folder_name + '\\' + date + '_' + str(i_file) + '.csv'
        total_result_df.to_csv(file_path, index=False)









