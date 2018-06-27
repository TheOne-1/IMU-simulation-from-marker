import xlwt
from datetime import datetime

file_path = '1.xls'
wb = xlwt.Workbook()
ws = wb.add_sheet('0')
wb.save(file_path)  # initialize the excel file