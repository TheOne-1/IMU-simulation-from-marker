# this file analyse and averages all the speed and output
import xlwt
from const import *
import numpy as np
import pandas as pd
from ResultPresenter import Presenter

file_date = '20180721'
result_file = RESULT_PATH + 'result_uni\\' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + 'result_uni\\' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')


for segment_moved in MOVED_SEGMENT_NAMES:

wb.save(file_path)


