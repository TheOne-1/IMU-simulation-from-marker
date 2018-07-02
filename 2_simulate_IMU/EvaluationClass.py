
import matplotlib.pyplot as plt
# from keras.callbacks import EarlyStopping
# from keras.models import *
import sklearn
from sklearn.metrics import r2_score
from const import *
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import os
from sklearn.decomposition import PCA
import json


class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test, y_column_names, base_model):
        self.__params_column_names = y_column_names
        self.__do_scaling = DO_SCALING
        self.__do_PCA = DO_PCA
        self.__base_model = base_model
        self.__batch_size = 50  # the size of data that be trained together
        self.__base_scaler = preprocessing.StandardScaler()
        self.__pca = PCA(n_components=N_COMPONENT)

        if self.__do_scaling:
            x_scalar = sklearn.clone(self.__base_scaler)
            y_scalar = sklearn.clone(self.__base_scaler)
            x_scalar.fit(x_train)
            y_scalar.fit(y_train)
            self.__x_train = x_scalar.transform(x_train)
            self.__y_train = y_scalar.transform(y_train)
            self.__x_test = x_scalar.transform(x_test)
            self.__y_test = y_scalar.transform(y_test)
        else:  # transfer dataframe to ndarray
            self.__x_train = x_train.as_matrix()
            self.__y_train = y_train.as_matrix()
            self.__x_test = x_test.as_matrix()
            self.__y_test = y_test.as_matrix()

        if self.__do_PCA:
            self.__pca.fit(self.__x_train)
            self.__x_train = self.__pca.transform(self.__x_train)
            self.__x_test = self.__pca.transform(self.__x_test)

        self.__models = []
        for i_model in range(self.__y_train.shape[1]):
            self.__models.append(sklearn.clone(self.__base_model))

    def set_x(self, x_train, x_test):
        if self.__do_scaling:
            x_scalar = sklearn.clone(self.__base_scaler)
            x_scalar.fit(x_train)
            self.__x_test = x_scalar.transform(x_test)
        else:
            self.__x_test = x_test.as_matrix()

        if self.__do_PCA:
            self.__x_test = self.__pca.transform(self.__x_test)

    def train_sklearn(self):
        i_output = 0
        for model in self.__models:
            model.fit(self.__x_train, self.__y_train[:, i_output])
            i_output += 1

    def evaluate_sklearn(self):
        scores = []
        i_output = 0
        for model in self.__models:
            result = model.predict(self.__x_test)
            scores.append(r2_score(self.__y_test[:, i_output], result))
            i_output += 1
        return scores

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
    # save the current result to the total result
    def store_current_result(score_list, xy_generator, total_result_columns):
        result_df = Evaluation.initialize_result_df(total_result_columns)
        segment_name = xy_generator.get_moved_segment()
        subject_id = xy_generator.get_subject_id()
        cylinder_diameter = xy_generator.get_cylinder_diameter()
        speed = xy_generator.get_speed()
        for score in score_list:
            score_numbers = score[0]
            if segment_name in ['trunk', 'pelvis']:
                x_offset, z_offset = score[1], score[2]
                y_offset, theta_offset = 0, 0
            else:
                theta_offset, z_offset = score[1], score[2]
                theta_radians = theta_offset * np.pi / 180     # change degree to radians
                x_offset = cylinder_diameter / 2 * (1 - np.cos(theta_radians))
                y_offset = cylinder_diameter / 2 * np.sin(theta_radians)
            result_item = [segment_name, subject_id, speed, x_offset, y_offset, z_offset, theta_offset]
            result_item.extend(score_numbers)
            df_item = pd.DataFrame([result_item], columns=total_result_columns)
            result_df = result_df.append(df_item)
        return result_df

    # save the dataframe and an introduction file
    @staticmethod
    def save_total_result(total_score_df, input_names, output_names, model):
        date = time.strftime('%Y%m%d')
        file_path = RESULT_PATH + 'result_' + date + '.csv'
        specification_txt_file = RESULT_PATH + 'result_' + date + '_specification.txt'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = RESULT_PATH + 'result_' + date + '_' + str(i_file) + '.csv'
            specification_txt_file = RESULT_PATH + 'result_' + date + '_' + str(i_file) + '_specification.txt'
        total_score_df.to_csv(file_path, index=False)

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


    # def train_nn(self, model):
    #     # lr = learning rate, the other params are default values
    #     optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #     model.compile(loss='mean_squared_error', optimizer=optimizer)
    #     # val_loss = validation loss, patience is the tolerance
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    #     # epochs is the maximum training round, validation split is the size of the validation set,
    #     # callback stops the training if the validation was not approved
    #     model.fit(self.__x_train, self.__y_train, batch_size=self.__batch_size,
    #               epochs=100, validation_split=0.2, callbacks=[early_stopping])
    #     self.__nn_model = model
    #
    # def evaluate_nn(self):
    #     result = self.__nn_model.predict(self.__x_test, batch_size=self.__batch_size)
    #     if self.__do_scaling:
    #         self.__y_test = self.__y_scalar.inverse_transform(self.__y_test)
    #         result = self.__y_scalar.inverse_transform(result)
    #
    #     score = r2_score(self.__y_test, result, multioutput='raw_values')
    #     return score












