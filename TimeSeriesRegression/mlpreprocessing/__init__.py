import numpy as np
import pandas as pd
import datetime

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


