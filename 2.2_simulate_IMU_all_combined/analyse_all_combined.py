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
file_date = '20180730'
folder_name = 'result_all_combined'
result_file = RESULT_PATH + folder_name + '\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

exp1_file = RESULT_PATH + 'result_segment\\20180730.csv'
exp1_result = pd.read_csv(exp1_file)

file_path = RESULT_PATH + folder_name + '\\' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')

Presenter.show_all_combined_result(result_df, exp1_result, ['FP1.ForX_NRMSE'], file_date, sub_num=1)
wb.save(file_path)
