# this file analyse and averages all the speed and output
import pandas as pd
import xlwt
from PresenterUni import Presenter
from const import *

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
]
file_date = '20180731'
folder_name = 'result_segment'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + folder_name + '\\' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')


for segment_moved in SEGMENT_NAMES:
    Presenter.show_segment_result(result_df[result_df['segment'] == segment_moved],
                                  ['ForX_R2', 'ForY_R2', 'ForZ_R2'], file_date)
    # Presenter.get_segment_force_matrix(result_df[result_df['segment'] == segment_moved], sheet)
wb.save(file_path)
