import pandas as pd
import xlwt
import numpy as np
from const import *

class ModelSelector:
    @staticmethod
    def store_impt_matrix(mean_matrice, column_names, row_names):
        # store mean matrice
        impt_matrix = pd.DataFrame(mean_matrice)
        impt_matrix.columns = column_names
        impt_matrix.insert(0, 'output_name', row_names)
        impt_matrix.to_csv(RESULT_PATH + 'importance_matrix\impt_matrice.csv', index=False)

    @staticmethod
    def store_std_matrix(mean_matrice, std_matrice, column_names, row_names):

        # store mean and std matrice
        std_matrix = pd.DataFrame(np.zeros([row_names.__len__(), column_names.__len__()]))
        for i_x in range(row_names.__len__()):
            for i_y in range(column_names.__len__()):
                text = str(round(mean_matrice[i_x, i_y], 3)) + ', ' + str(round(std_matrice[i_x, i_y], 3))
                std_matrix.iloc[i_x, i_y] = text
        std_matrix.columns = column_names
        std_matrix.insert(0, 'output_name', row_names)
        std_matrix.to_csv(RESULT_PATH + 'importance_matrix\std_matrice.csv', index=False)

    @staticmethod
    def store_impt_matrix_trial(trial_matrice, column_names, row_names):
        all_df = pd.DataFrame(trial_matrice[:, :, 0])
        all_df.columns = column_names
        all_df.insert(0, 'output_name', row_names)
        for i_matrix in range(1, trial_matrice.shape[2]):
            trial_df = pd.DataFrame(trial_matrice[:, :, i_matrix])
            trial_df.columns = column_names
            trial_df.insert(0, 'output_name', row_names)
            # all_df = all_df.append(empty_row, ignore_index=True)
            all_df.loc[-1] = ''
            all_df = all_df.append(trial_df, ignore_index=True)
        # store mean matrice
        all_df.to_csv(RESULT_PATH + 'importance_matrix\impt_matrice_all.csv', index=False)
