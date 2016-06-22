from mltools import rolling_univariate_window, build_rolling_window_dataset, verify_window_dataset, create_rondomsearch_configs4DL
import numpy as np
import pandas as pd
import datetime
from mltools import calculate_rmsle


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
        new_data.append([daskey, dict[daskey] if dict.has_key(daskey) else 0])

    return np.row_stack(new_data)






    #df.set_index(date_col_name,drop=True,inplace=True)
    #df.index = pd.DatetimeIndex(df.index)
    #d = datetime.now().date()
    #d2 = d - timedelta(days = days_back)
    #idx = pd.date_range(d2, d, freq = "D")
    #df = df.reindex(idx,fill_value=fill_value)
    #df[date_col_name] = pd.DatetimeIndex(df.index)

    return df

#list = np.array(range(100))

#data = build_rolling_window_dataset(list, 7)
#print(data)

#configs = create_rondomsearch_configs4DL((1,2,3), (5,10,15,20), (0.1, 0.2, 0.4),
#                                        (0, 0.01, 0.001), (0.01, 0.001, 0.0001), 50)

#for c in configs:
#    print c.tostr()

#appleSotcksDf = pd.read_csv('./data/USDvsEUExchangeRate.csv')
#print appleSotcksDf.head(20)

#fixedData = fill_in_missing_dates(appleSotcksDf, 'day', 'ExchangeRate')

#np.savetxt('USDvsEUExchangeRateFixed.csv', fixedData, fmt='%s', delimiter=',', header="day,ExchangeRate")
#print fixedData[:100]

#appleSotcksDf = pd.read_csv('./data/applestocksfixed.csv')
#appleSotcksDf = appleSotcksDf.fillna(method='pad')
#np.savetxt('temp.csv', appleSotcksDf, fmt='%s', delimiter=',', header="Date,Close")

df = pd.read_csv('forecastdata.csv')

print calculate_rmsle(df['RFR'].values, df['actual'].values)

