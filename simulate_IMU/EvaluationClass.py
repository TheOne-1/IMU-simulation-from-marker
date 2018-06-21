
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from keras.models import *
from sklearn import preprocessing


class Evaluation:

    def __init__(self, x_train, x_test, y_train, y_test, y_column_names, do_scaling=True, sub_num=10, gait_num=1):
        self.__x_training = x_train
        self.__x_testing = x_test
        self.__y_training = y_train
        self.__y_testing = y_test
        self.__gait_num = gait_num
        self.__sub_num = sub_num
        self.__train_set_len = 0
        self.__test_set_len = 0
        self.__params_column_names = y_column_names
        self.__do_scaling = do_scaling
        self.__nn_model = None
        self.__batch_size = 50  # the size of data that be trained together

        if self.__do_scaling:
            self.__x_scalar = preprocessing.StandardScaler().fit(self.__x_training)
            self.__y_scalar = preprocessing.StandardScaler().fit(self.__y_training)
            self.__x_training = self.__x_scalar.transform(self.__x_training)
            self.__y_training = self.__y_scalar.transform(self.__y_training)
            self.__x_testing = self.__x_scalar.transform(self.__x_testing)
            self.__y_testing = self.__y_scalar.transform(self.__y_testing)
        else:  # transfer dataframe to ndarray
            self.__x_training = self.__x_training.as_matrix()
            self.__y_training = self.__y_training.as_matrix()
            self.__x_testing = self.__x_testing.as_matrix()
            self.__y_testing = self.__y_testing.as_matrix()

    def set_testing_data(self, x, y, x_scalar, y_scalar):
        self.__x_testing = x
        self.__y_testing = y
        self.__test_set_len = int(self.__x_testing.shape[0] / self.__gait_num / self.__sub_num)
        self.__x_scalar = x_scalar
        self.__y_scalar = y_scalar

        if self.__do_scaling:
            self.__x_testing = self.__x_scalar.transform(self.__x_testing)
            self.__y_testing = self.__y_scalar.transform(self.__y_testing)
        else:  # transfer dataframe to ndarray
            self.__x_testing = self.__x_testing.as_matrix()
            self.__y_testing = self.__y_testing.as_matrix()

    def train_nn(self, model):
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        model.fit(self.__x_training, self.__y_training, batch_size=self.__batch_size,
                  epochs=100, validation_split=0.2, callbacks=[early_stopping])
        self.__nn_model = model

    def evaluate_nn(self, trial_name):
        result = self.__nn_model.predict(self.__x_testing, batch_size=self.__batch_size)
        if self.__do_scaling:
            self.__y_testing = self.__y_scalar.inverse_transform(self.__y_testing)
            result = self.__y_scalar.inverse_transform(result)

        score = r2_score(self.__y_testing, result, multioutput='raw_values')
        for i_plot in range(result.shape[1]):
            plt.figure()
            plt.plot(self.__y_testing[:, i_plot], 'b', label='true value')
            plt.plot(result[:, i_plot], 'r', label='predicted value')
            plt.title(trial_name + '  ' + self.__params_column_names[i_plot] + '  R2: ' + str(score[i_plot])[0:5])
            plt.legend()
            for i_subject in range(0, self.__sub_num):
                for i_gait in range(1, self.__gait_num):
                    line_x_gait = self.__test_set_len * i_gait + i_subject * (self.__test_set_len * self.__gait_num)
                    plt.plot((line_x_gait, line_x_gait), (-0.5, 0.5), 'y--')
                if i_subject != 0:
                    line_x_sub = self.__x_testing.shape[0] / self.__sub_num * i_subject
                    plt.plot((line_x_sub, line_x_sub), (-0.5, 0.5), 'black')

    def train_sklearn(self, model):
        model.fit(self.__x_training, self.__y_training)
        self.__sklearn_model = model

    def evaluate_sklearn(self, show_plot=False):
        result = self.__sklearn_model.predict(self.__x_testing)
        score = r2_score(self.__y_testing, result)
        if show_plot:
            if self.__do_scaling:
                self.__y_testing = self.__y_scalar.inverse_transform(self.__y_testing)
                result = self.__y_scalar.inverse_transform(result)
            # plot
            for i_plot in range(result.shape[1]):
                plt.figure()
                plt.plot(self.__y_testing[:, i_plot], 'b', label='true value')
                plt.plot(result[:, i_plot], 'r', label='predicted value')
                plt.title(self.__params_column_names[i_plot] + '  R2: ' + str(score[i_plot])[0:5])
                plt.legend()
                for i_subject in range(0, self.__sub_num):
                    for i_gait in range(1, self.__gait_num):
                        line_x_gait = self.__test_set_len * i_gait + i_subject * (self.__test_set_len * self.__gait_num)
                        plt.plot((line_x_gait, line_x_gait), (-0.5, 0.5), 'y--')
                    if i_subject != 0:
                        line_x_sub = self.__x_testing.shape[0] / self.__sub_num * i_subject
                        plt.plot((line_x_sub, line_x_sub), (-0.5, 0.5), 'black')
        return score
