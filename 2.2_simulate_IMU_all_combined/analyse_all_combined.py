# this file analyse and averages all the speed and output
import pandas as pd
import xlwt

from PresenterUni import Presenter
from const import *



Presenter.show_all_combined_result(result_df, exp1_result, ['FP1.ForX_NRMSE'], file_date, sub_num=1)
