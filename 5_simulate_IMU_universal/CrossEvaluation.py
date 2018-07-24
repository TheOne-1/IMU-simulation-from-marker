
import matplotlib.pyplot as plt
# from keras.callbacks import EarlyStopping
# from keras.models import *
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
from sklearn.model_selection import KFold

class CrossValidation:
    def __init__(self, y_column_names, base_model, n_fold):
        self.__params_column_names = y_column_names
        self.__base_model = base_model
        self.__batch_size = 50  # the size of data that be trained together
        self.__base_scaler = preprocessing.StandardScaler()
        self.__pca = PCA(n_components=N_COMPONENT)
        self.__n_fold = n_fold
        self.__k_fold = KFold(n_fold, shuffle=True, random_state=0)

    def set_original_xy(self, x_original, y_original):
        x_original = x_original.copy()
        y_original = y_original.copy()
        if DO_SCALING:
            self.__x_scalar = sklearn.clone(self.__base_scaler)
            self.__y_scalar = sklearn.clone(self.__base_scaler)
            x_original = self.__x_scalar.fit_transform(x_original)
            y_original = self.__y_scalar.fit_transform(y_original)
        if DO_PCA:
            self.__pca.fit(x_original)
            x_original = self.__pca.transform(x_original)
        if DO_SHUFFLING:
            x_original, y_original = shuffle(x_original, y_original)
        self.__x_original = x_original
        self.__y_original = y_original
        self.__models = []
        for i_model in range(self.__y_original.shape[1]):
            for i_fold in range(self.__n_fold):
                self.__models.append(sklearn.clone(self.__base_model))

    def set_modified_x(self, x_modified):
        x_modified = x_modified.copy()
        if DO_SCALING:
            x_modified = self.__x_scalar.transform(x_modified)
        if DO_PCA:
            x_modified = self.__pca.transform(x_modified)
        self.__x_modified = x_modified

    def train_cross_validation(self):
        i_model = 0
        for i_output in range(self.__y_original.shape[1]):
            for i_fold in range(self.__n_fold):
                train_indices, test_indices = self.__k_fold.split(self.__x_original, groups=i_fold)
                x_train = self.__x_original[train_indices]
                y_train = self.__y_original[train_indices]
                self.__models[i_model].fit(x_train, y_train[:, i_output])
                i_model += 1

    def test_get_scores(self):
        i_model = 0
        scores = []
        score_fold = np.zeros([self.__n_fold])
        for i_output in range(self.__y_original.shape[1]):
            for i_fold in range(self.__n_fold):
                train_indices, test_indices = self.__k_fold.split(self.__x_original, groups=i_fold)
                x_test = self.__x_modified[test_indices]
                y_test = self.__y_original[test_indices]
                y_pred = self.__models[i_model].predict(x_test)
                score_fold[i_fold] = r2_score(y_test, y_pred)
            scores.append(np.mean(score_fold))

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
    def save_result(total_result_df, model, input_names, output_names, folder_name):
        date = time.strftime('%Y%m%d')
        file_path = RESULT_PATH + folder_name + '\\' + date + '.csv'
        specification_txt_file = RESULT_PATH + folder_name + '\\' + date + '_specification.txt'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = RESULT_PATH + folder_name + '\\' + date + '_' + str(i_file) + '.csv'
            specification_txt_file = RESULT_PATH + folder_name + '\\' + date + '_' + str(i_file) + '_specification.txt'
        total_result_df.to_csv(file_path, index=False)

        # write a specification file about details
        input_str = ', '.join(input_names)
        output_str = ', '.join(output_names)

        if DO_SCALING:
            scaling_str = 'Scaling: StandardScaler'
        else:
            scaling_str = 'Scaling: None'
        if DO_PCA:
            pca_str = 'Feature selection: PCA, n component = ' + str(N_COMPONENT)
        else:
            pca_str = 'Feature selection: None'

        content = 'Machine learning model: ' + model.__class__.__name__ + '\n' + \
                  'Model parameters: ' + json.dumps(model.get_params()) + '\n' + \
                  'Input: ' + input_str + '\n' + \
                  'Output: ' + output_str + '\n' + scaling_str + '\n' + pca_str + '\n'

        with open(specification_txt_file, 'w') as file:
            file.write(content)











