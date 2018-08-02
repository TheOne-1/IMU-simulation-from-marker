import pandas as pd


file = 'D:\Tian\Research\Projects\ML Project\gait_database\GaitDatabase\data\T049\mocap-049.txt'
# f = open(file)
# text = f.readline()

df = pd.read_csv(file, sep='\t')

column_names = list(df.columns)
column_names[3] = 'mmm'
df.columns = column_names
df.to_csv('rrr', sep='\t', index=False)
x = 1