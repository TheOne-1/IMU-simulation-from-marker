import pandas as pd
from Presenter import Presenter
from const import *

file_date = '20180806'
folder_name = 'result_segment_translation'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

Presenter.get_acceptable_area('NRMSE', file_date, threshold=0.25)


