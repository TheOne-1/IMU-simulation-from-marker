# this file analyse and averages all the speed and output
import pandas as pd
import xlwt

from Presenter import Presenter
from const import *


file_date = '20181008'
folder_name = 'result_all_trans_rota'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)


Presenter.show_all_combined_result(result_df, file_date, folder_name, model_name='Gradient Boosting')
