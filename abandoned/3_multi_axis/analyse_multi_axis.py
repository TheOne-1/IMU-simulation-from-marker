# this file analyse and averages all the speed and output
import pandas as pd
import xlwt

from abandoned.ResultPresenter import Presenter
from const import *

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
]
file_date = '20180725'
folder_name = 'result_multi_axis'

result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + folder_name + '\\' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')

Presenter.show_result_multi_axis(result_df, ['r_thigh_theta_offset', 'l_shank_z_offset'], 'FP1.ForZ')
wb.save(file_path)
