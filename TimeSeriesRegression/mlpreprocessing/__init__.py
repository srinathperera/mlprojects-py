import numpy as np
import pandas as pd
import datetime


import feather


#extract dates and other column from data frame and fill missing entries as None
def fill_in_missing_dates(df, date_col_name, other_col):
    startd = df[date_col_name].values[0]
    endd = df[date_col_name].values[-1]
    print startd, endd
    idx = pd.date_range(startd, endd)

    dict = {}
    for index, row in df.iterrows():
        dict[row[date_col_name]] = row[other_col]

    new_data = []
    for d in idx:
        pydate = d.to_pydatetime()
        daskey = pydate.strftime('%Y-%m-%d')
        new_data.append([daskey, dict[daskey] if dict.has_key(daskey) else None])

    return np.row_stack(new_data)


def pad_missing_value(df):
    return df.fillna(method='pad')

#feather is a fast binary format https://blog.rstudio.org/2016/03/29/feather/
def feather2df(path):
    return feather.read_dataframe(path)

def df2feather(df, path):
    return feather.write_dataframe(df,path)

##Numpy binary format http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html#numpy.save


#HDF store http://www.pytables.org/cookbook/inmemory_hdf5_files.html
#The HDF5 library provides functions to allow an application to work with a file in memory for faster reads and writes. File contents are kept in memory until the file is closed. At closing, the memory version of the file can be written back to disk or abandoned.
#http://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas
#http://pandas-docs.github.io/pandas-docs-travis/io.html#hdf5-pytables
#http://scikit-learn.org/stable/modules/scaling_strategies.html
#install py tables ( bit complicated http://docs.h5py.org/en/latest/build.html)
