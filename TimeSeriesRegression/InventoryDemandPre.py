import numpy as np
import pandas as pd

from mltoo


appleSotcksDf = pd.read_csv('./data/applestocksfixed.csv')
appleSotcksDf = appleSotcksDf.fillna(method='pad')
np.savetxt('temp.csv', appleSotcksDf, fmt='%s', delimiter=',', header="Date,Close")
