# this file analyse and averages all the speed and output
import pandas as pd
from PresenterUni import Presenter
from const import *

file_date = '20180806'
folder_name = 'result_segment_translation'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

Presenter.save_segment_translation_result(result_df, file_date)
Presenter.show_segment_translation_result('NRMSE', file_date)
