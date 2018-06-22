
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from keras.models import *
from sklearn import preprocessing
import pandas as pd


class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test, y_column_names, do_scaling=True):
        self.__params_column_names = y_column_names
        self.__do_scaling = do_scaling
        self.__nn_model = None
        self.__sklearn_model = None
        self.__batch_size = 50  # the size of data that be trained together

        if self.__do_scaling:
            self.__x_scalar = preprocessing.StandardScaler().fit(x_train)
            self.__y_scalar = preprocessing.StandardScaler().fit(y_train)
            self.__x_train = self.__x_scalar.transform(x_train)
            self.__y_train = self.__y_scalar.transform(y_train)
            self.__x_test = self.__x_scalar.transform(x_test)
            self.__y_test = self.__y_scalar.transform(y_test)
        else:  # transfer dataframe to ndarray
            self.__x_train = x_train.as_matrix()
            self.__y_train = y_train.as_matrix()
            self.__x_test = x_test.as_matrix()
            self.__y_test = y_test.as_matrix()

    def set_x_test(self, x_test):
        if self.__do_scaling:
            self.__x_test = self.__x_scalar.transform(x_test)
        else:
            self.__x_test = x_test.as_matrix()

    def train_nn(self, model):
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        model.fit(self.__x_train, self.__y_train, batch_size=self.__batch_size,
                  epochs=100, validation_split=0.2, callbacks=[early_stopping])
        self.__nn_model = model

    def evaluate_nn(self):
        result = self.__nn_model.predict(self.__x_test, batch_size=self.__batch_size)
        if self.__do_scaling:
            self.__y_test = self.__y_scalar.inverse_transform(self.__y_test)
            result = self.__y_scalar.inverse_transform(result)

        score = r2_score(self.__y_test, result, multioutput='raw_values')

    def train_sklearn(self, model):
        model.fit(self.__x_train, self.__y_train)
        self.__sklearn_model = model

    def evaluate_sklearn(self):
        result = self.__sklearn_model.predict(self.__x_test)
        score = r2_score(self.__y_test, result)
        return score

    # 修改的简单些
    @staticmethod
    def show_result(score_list, xy_generator):
        axis_1_range, axis_2_range = xy_generator.get_point_range()
        # change result as an image
        result_im = np.zeros([axis_2_range.__len__(), axis_1_range.__len__()])
        i_y, i_result = 0, 0
        for i_x in range(axis_1_range.__len__()):
            for i_y in range(axis_2_range.__len__()):
                # for image, row and column are exchanged compared to ndarray
                result_im[i_y, i_x] = score_list[i_result][0]
                i_result += 1
        fig, ax = plt.subplots()
        plt.imshow(result_im)
        x_label = list(axis_1_range)
        ax.set_xticks(range(result_im.shape[1]))
        ax.set_xticklabels(x_label)
        y_label = list(axis_2_range)
        ax.set_yticks(range(result_im.shape[0]))
        ax.set_yticklabels(y_label)
        plt.show()

    @staticmethod
    def save_result(score_list, xy_generator):
        i_pos = 0
        result = np.zeros([score_list.__len__(), 3])
        for score in score_list:
            result[i_pos, :] = [score[0], score[1], score[2]]
            i_pos += 1

        result_df = pd.DataFrame(result)
        result_df.columns = ['score', 'x', 'y']
        result_df.to_csv('D:\Tian\Research\Projects\ML Project\simulatedIMU\python\\0517GaitDatabase\data_' +
                         xy_generator.get_moved_segment() + '.csv')














