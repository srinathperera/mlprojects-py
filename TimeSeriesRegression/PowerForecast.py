import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing
import math

from scipy import stats
from tsforecasttools import run_timeseries_froecasts, aggregate_hl
from sklearn.linear_model import LinearRegression


from mltools import build_rolling_window_dataset, l2norm, regression_with_GBR, regression_with_LR, regression_with_RFR
from mltools import train_test_split,print_graph_test,almost_correct_based_accuracy
from mltools import regression_with_dl, print_regression_model_summary, report_scores

from datetime import datetime


#df = pd.read_csv("household_power_consumption200k.txt", delimiter=';')
#print(df.head())
#power_data = df['Global_active_power'].values
#print (power_data)


#with open("household_power_consumption200k.txt") as f:
with open("household_power_consumption.txt") as f:
    data = csv.reader(f, delimiter=";")
    power_data = []
    cycle_data = []
    for line in data:
        try:
            power_data.append(float(line[2]))
            d = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            d = datetime.strptime(line[0] + " " + line[1], '%d/%m/%Y %H:%M:%S')
            cycle_data.append([d.month, d.weekday(), d.hour])
            #d.year
        except ValueError:
            pass

normalizing_factor = l2norm(power_data)

power_data = preprocessing.normalize(power_data, norm='l2')[0]


window_size = 7


#build rolling window. It is of size row_count - window_size ( due to first window_size -1 due to lack of data, and last value
#due to lack of training (Y) data
X_all, Y_all = build_rolling_window_dataset(power_data, window_size)

#X_all = X_all[:10000,]
#Y_all = Y_all[:10000]

row_count = X_all.shape[0];
training_set_size = int(0.7*row_count)

print("X_all.shape", X_all.shape)

#http://docs.scipy.org/doc/scipy/reference/stats.html
print "z scores"
zscore_vals = []
entropy_vals = []
for i in range(len(X_all)):
    z = stats.mstats.zscore(X_all[i])[3]
    if math.isnan(z):
        z = 0
    #print X_all[i], z
    zscore_vals.append(z)
    entropy_vals.append(stats.entropy(X_all[i]))

mavg1_vals = []
for i in range(len(X_all)):
    mavg1_vals.append(np.mean(X_all[i]))


mavg2_vals = aggregate_hl(mavg1_vals, 2)
mavg4_vals = aggregate_hl(mavg1_vals, 4)
mavg8_vals = aggregate_hl(mavg1_vals, 8)
mavg16_vals = aggregate_hl(mavg1_vals, 16)


#print "regress line"
#regress_val = []
#for i in range(len(X_all)):
#    slope, intercept, r_value, p_value, std_err = stats.linregress(range(4),X_all[i])
#    #print slope
#    regress_val.append(slope)


#zscore_vals = [ stats.mstats.zscore(X_all[i]) for i in range(len(X_all)) ]
#cos_vals = [ math.cos(i*2*math.pi/7) for i in range(len(X_all)) ]
print "done"

#lr = LinearRegression(normalize=True)
#lr.fit(X_all, Y_all)
#y_pred_lr = lr.predict(X_all)

print ("cycle_data.head", cycle_data[:10])

cycle_data = np.array(cycle_data[7:])
print("X_all.shape", X_all.shape)
print("cycle_data.shape", cycle_data.shape)

#X_all = np.column_stack((X_all, cycle_data, zscore_vals, entropy_vals, mavg1_vals, mavg2_vals, mavg4_vals, mavg8_vals, mavg16_vals))
#X_all = np.column_stack((X_all, cycle_data))
#X_all = np.column_stack((cycle_data, zscore_vals, entropy_vals, mavg1_vals, mavg2_vals, mavg4_vals, mavg8_vals, mavg16_vals))

#X_all = np.column_stack((cycle_data, zscore_vals, entropy_vals, mavg1_vals, mavg2_vals, mavg4_vals, mavg8_vals, mavg16_vals))

X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

print ("normalizing_factor", normalizing_factor)
#run_timeseries_froecasts(X_train, y_train, X_test, y_test, window_size, epoch_count=10)
#regression_with_GBR(X_train, y_train, X_test, y_test)
regression_with_LR(X_train, y_train, X_test, y_test)
#regression_with_RFR(X_train, y_train, X_test, y_test)

nodes_in_layer = 500
number_of_hidden_layers = 2
droput = 0.05
#droput = 0.1
activation_fn='relu'
#y_pred_dl = regression_with_dl(X_train, y_train, X_test, y_test, nodes_in_layer,
#                      number_of_hidden_layers, droput, activation_fn, 10)
#print_regression_model_summary("DL", y_test, y_pred_dl)

#np.savetxt('pforecast.csv', np.column_stack((X_all, Y_all)), delimiter=',', header="w1,w2,w3,w4,w5,w6,w7,year,month,weekday,hour,target")   # X is an array


