# this file analyse and averages all the speed and output
import pandas as pd
import xlwt
from PresenterUni import Presenter
from const import *

file_date = '20180731'
folder_name = 'result_segment'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

Presenter.show_segment_result_all('RMSE', file_date)


#
# for segment_moved in SEGMENT_NAMES:
#     Presenter.show_segment_result(result_df[result_df['segment'] == segment_moved],
#                                   ['ForX_R2', 'ForY_R2', 'ForZ_R2'], file_date)