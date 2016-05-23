from mltools import rolling_univariate_window, build_rolling_window_dataset, verify_window_dataset
import numpy as np

list = np.array(range(100))

data = build_rolling_window_dataset(list, 7)
print(data)