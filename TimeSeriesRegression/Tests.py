from mltools import rolling_univariate_window, build_rolling_window_dataset, verify_window_dataset, create_rondomsearch_configs4DL
import numpy as np

list = np.array(range(100))

#data = build_rolling_window_dataset(list, 7)
#print(data)

configs = create_rondomsearch_configs4DL((1,2,3), (5,10,15,20), (0.1, 0.2, 0.4),
                                        (0, 0.01, 0.001), (0.01, 0.001, 0.0001), 50)

for c in configs:
    print c.tostr()
