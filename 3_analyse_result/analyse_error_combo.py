# this file try to analyze the relationship between error directions
# we want to know whether different error source are independent
import xlwt
from const import *
import numpy as np
import pandas as pd
from ResultPresenter import Presenter

file_date = '20180627'
result_file = RESULT_PATH + 'result_' + file_date + '.csv'
result_df = pd.read_csv(result_file)

file_path = RESULT_PATH + 'result_' + file_date + '\\result_' + file_date + '_matrix.xls'
wb = xlwt.Workbook()
sheet = wb.add_sheet('Sheet1')


for segment_moved in MOVED_SEGMENT_NAMES:
    Presenter.independence_analysis(result_df[result_df['segment'] == segment_moved])
wb.save(file_path)

