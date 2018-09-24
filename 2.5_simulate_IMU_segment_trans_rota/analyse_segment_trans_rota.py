# this file analyse and averages all the speed and output
import pandas as pd
from PresenterTransRota import PresenterTransRota
from const import *

file_date = '20180830'
folder_name = 'result_segment_trans_rota'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

# PresenterTransRota.save_segment_trans_rota(result_df, file_date)
PresenterTransRota.show_segment_trans_rota('NRMSE', file_date, model_name='Gradient Boosting')
