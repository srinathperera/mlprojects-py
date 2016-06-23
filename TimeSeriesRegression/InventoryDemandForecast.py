import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


from tsforecasttools import run_timeseries_froecasts, regression_with_RFR
from mltools import preprocess2DtoZeroMeanUnit, preprocess1DtoZeroMeanUnit, train_test_split, print_feature_importance
from mltools import calculate_rmsle, almost_correct_based_accuracy

import sys
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

def addFeildStatsAsFeatures(df, feild_name, testDf, drop=False, default_mean=None, default_stddev=None):
    groupData = df.groupby([feild_name])['Demanda_uni_equil']
    meanData = groupData.mean()
    stddevData = groupData.std()

    df[feild_name+"_Mean"] = [meanData.get(x, default_mean) for x in df[feild_name]]
    df[feild_name+"_StdDev"] = [stddevData.get(x, default_stddev) for x in df[feild_name]]
    if drop:
        newDf = df.drop(feild_name,1)

    if testDf is not None:
        testDf[feild_name+"_Mean"] = [meanData.get(x, default_mean) for x in testDf[feild_name]]
        testDf[feild_name+"_StdDev"] = [stddevData.get(x, default_stddev) for x in testDf[feild_name]]
        if drop:
            testDf = testDf.drop(feild_name,1)

    return newDf, testDf



df = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/trainitems300.csv')
#df = pd.read_csv('data/train.csv')


#testDf = pd.read_csv('/Users/srinath/playground/data-science/BimboInventoryDemand/test.csv')
#testDf = testDf[(testDf['Producto_ID'] <= 300)]
testDf = None

#print "shapes train, test", df.shape, testDf.shape
print "shapes train, test", df.shape

demand_val_mean = df['Demanda_uni_equil'].mean()
demand_val_stddev = df['Demanda_uni_equil'].mean()

#calculate average slope
grouped = df.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil']
def avgdiff(group):
    group = group.values
    if len(group) > 1:
        return np.mean([group[i] - group[i-1] for i in range(1, len(group))])
    else:
        return 0

slopeMap = grouped.apply(avgdiff)
groupedMeanMap = grouped.mean()


slopes = []
groupedMeans = []
for index, row in df.iterrows():
    slopes.append(slopeMap[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])
    groupedMeans.append(groupedMeanMap[row['Agencia_ID'], row['Canal_ID'], row['Ruta_SAK'], row['Cliente_ID'], row['Producto_ID']])

df['Slopes']= slopes
df['groupedMeans']= groupedMeans

#add mean and stddev
df, testDf = addFeildStatsAsFeatures(df,'Agencia_ID', testDf, drop=True)
df, testDf = addFeildStatsAsFeatures(df,'Canal_ID', testDf, drop=True)
df, testDf = addFeildStatsAsFeatures(df,'Ruta_SAK', testDf, drop=True)
df, testDf = addFeildStatsAsFeatures(df,'Cliente_ID', testDf, drop=True)
df, testDf = addFeildStatsAsFeatures(df,'Producto_ID', testDf, True, demand_val_mean, demand_val_stddev)



print df.head(10)

#TODO explore this more http://pandas.pydata.org/pandas-docs/stable/missing_data.html
df = df.fillna(0)

y_actual = df['Demanda_uni_equil'].values
Y_all, parmsFromNormalization = preprocess1DtoZeroMeanUnit(y_actual)
df = df.drop('Demanda_uni_equil',1)
df = df.drop('Venta_uni_hoy',1)
df = df.drop('Venta_hoy',1)
df = df.drop('Dev_uni_proxima',1)
df = df.drop('Dev_proxima',1)

print "df", list(df)
#print "testDf", list(testDf)


X_all = preprocess2DtoZeroMeanUnit(df.values.copy())

training_set_size = int(0.7*X_all.shape[0])
X_train, X_test, y_train, y_test = train_test_split(training_set_size, X_all, Y_all)

#run_timeseries_froecasts(X_train, y_train, X_test, y_test, -1, 10, parmsFromNormalization)
#regression_with_RFR(X_train, y_train, X_test, y_test, parmsFromNormalization)


rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

#predict data set
y_pred_rfr = rfr.predict(X_test)

print_feature_importance(X_test, y_test,rfr.feature_importances_)

#undo the normalization
y_actual = y_actual[-1*len(y_pred_rfr):]
y_pred_corrected = (y_pred_rfr*parmsFromNormalization.std*parmsFromNormalization.sqrtx2) + parmsFromNormalization.mean

error_AC, rmsep, mape, rmse = almost_correct_based_accuracy(y_actual, y_pred_corrected, 10)
rmsle = calculate_rmsle(y_actual, y_pred_corrected)
print ">> %s AC_errorRate=%.1f RMSEP=%.6f MAPE=%6f RMSE=%6f rmsle=%.5f" %("RFR", error_AC, rmsep, mape, rmse, rmsle)

if testDf is not None:
    temp = testDf.drop('id',1)
    temp = temp.fillna(0)
    print list(temp)
    kaggale_test = preprocess2DtoZeroMeanUnit(temp.values.copy())
    kaggale_predicted = rfr.predict(kaggale_test)

    to_save = np.column_stack((np.array(list(range(len(kaggale_predicted)))), kaggale_predicted))
    np.savetxt('submission.csv', to_save, delimiter=',', header="id,Demanda_uni_equil")   # X is an array
