
import pandas as pd
from Presenter import Presenter
from const import *

file_date = '20180821'
folder_name = 'result_segment_rotation'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

Presenter.save_segment_rotation_result(result_df, file_date)
Presenter.show_segment_rotation_result(file_date)
