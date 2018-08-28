# this file only analysis selected output and speed
import pandas as pd
import xlwt

from Presenter import Presenter
from const import *

# the following output will be averaged
output_names = [
    # 'FP1.ForX',
    # 'FP2.ForX',
    'FP1.ForY',
    # 'FP2.ForY',
    # 'FP1.ForZ',
    # 'FP2.ForZ',
    # 'FP1.CopX',
    # 'FP1.CopY',
    # 'FP2.CopX',
    # 'FP2.CopY'
]
# the following speeds will be averaged
speed_names = [
    '0.8'
    # '1.2'
    # '1.6'
]

file_date = '20180704'
result_file = RESULT_PATH + 'result_' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + 'result_' + file_date + '\\result_' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')

for segment_moved in SEGMENT_NAMES:
    Presenter.show_selected_result(result_df[result_df['segment'] == segment_moved], file_date, output_names, speed_names)
    # Presenter.get_selected_matrix(result_df[result_df['segment'] == segment_moved], sheet, output_names, speed_names)
    # Presenter.get_segment_cop_matrix(result_df[result_df['segment'] == segment_moved], sheet)
wb.save(file_path)















