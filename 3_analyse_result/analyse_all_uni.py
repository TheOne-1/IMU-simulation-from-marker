# this file analyse and averages all the speed and output
import xlwt
from const import *
import numpy as np
import pandas as pd
from ResultPresenter import Presenter

output_names = [
    'FP1.ForX',
    'FP2.ForX',
    'FP1.ForY', 'FP2.ForY',
    'FP1.ForZ', 'FP2.ForZ',
    'FP1.CopX', 'FP1.CopY',
    'FP2.CopX', 'FP2.CopY'
]
file_date = '20180721'
result_file = RESULT_PATH + 'result_uni\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + 'result_uni\\' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')

Presenter.show_combined_result(result_df, ['l_shank_theta_offset', 'l_shank_z_offset'], 'FP1.ForZ')
wb.save(file_path)
