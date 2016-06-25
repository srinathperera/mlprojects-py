import pandas as pd
import xgboost as xgb


#http://datascience.stackexchange.com/questions/9483/xgboost-linear-regression-output-incorrect
#http://xgboost.readthedocs.io/en/latest/get_started/index.html
#https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost

df = pd.DataFrame({'x':[1,2,3], 'y':[10,20,30]})
X_train = df.drop('y',axis=1)
Y_train = df['y']
T_train_xgb = xgb.DMatrix(X_train, Y_train)

params = {"objective": "reg:linear", "booster":"gblinear"}
gbm = xgb.train(dtrain=T_train_xgb,params=params)
Y_pred = gbm.predict(xgb.DMatrix(pd.DataFrame({'x':[4,5]})))
print Y_pred