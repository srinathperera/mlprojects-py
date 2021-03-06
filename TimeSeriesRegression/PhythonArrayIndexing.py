import numpy as np
from scipy import stats
from sklearn.utils import shuffle

from mltools import preprocess1DtoZeroMeanUnit, undoPreprocessing

#http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial.html
#https://courses.p2pu.org/he/groups/scientific-python/content/reshaping-arrays/

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#All arrays generated by basic slicing are always views of the original array.

#- values wrap around
print(x[-1])

print("x[2:]", x[2:])

#here i is the starting index, j is the stopping index, and k is the step
print(x[2:8:1])


xx = np.array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])

print("xx[0,0]",xx[0,0])

print("xx[0:2,:]",xx[0:2,:])

print("x[:, :-1]", xx[:, :-1])


print(list(range(1, 100, 1)))


a = np.array([0, 1, 2, 3, 4])
b = np.array([10, 11, 12, 13, 14])

print np.vstack((a,b))
print np.hstack((a,b))

#c = a.append(b).reshape(5,2)
#print(c)


print ("xx[:, :-1]",xx[:, :-1])
print ("xx[:, -1]",xx[:, -1])


def transform_dataset4rnn(dataset, sequence_length, prediction_index):
    result = []
    dataset_length = dataset.shape[1]
    for index in range(dataset_length - sequence_length):
        result.append(dataset[:, index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)

    print("result",result)
    print("result.shape", result.shape)

    #result_mean = result.mean()
    #result -= result_mean
    #print "Shift : ", result_mean
    #print "Data  : ", result.shape

    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :, :]

    #np.random.shuffle(train)
    #print("train", train.shape)
    #print("train", train)

    X_train = train[:, :-1, :]
    y_train = train[:, -1, prediction_index]
    X_test = result[row:, :-1]
    y_test = result[row:, -1, prediction_index]
    return [X_train, y_train, X_test, y_test]
    #return [[], [], [], []]

print("=========================")

data_size = 10
power1 = [x * 1.0 for x in range(0, data_size)]
power2 = [x * 10.0 for x in range(0, data_size)]


dataset = np.array([power1, power2])
print("dataset", dataset)
X_train, y_train, X_test, y_test =  transform_dataset4rnn(dataset, 3, 0)

print("y_test", y_test)
print("X_test", X_test)

print("y_train", y_train)
print("X_train",X_train)

#xx = np.random.rand(10,4)

#for i in range(len(xx)):
#    z = stats.mstats.zscore(xx[i])[3]
#    print xx[i], z

#for i in range(len(xx)):
#    slope, intercept, r_value, p_value, std_err = stats.linregress(range(4),xx[i])
#    print slope

#regress_val = [ stats.linregress(range(4),xx[i]) for i in range(len(xx)) ]
#print zscore_vals

xx = np.random.rand(10,4)

a = []
for i in range(10):
    a.append([i, i+1, i+2])

npa = np.array(a)

print("xx.shape", xx.shape, "npa.shape", npa.shape)

X_all = np.column_stack((xx, npa))

print("X_all.shape",X_all.shape)

#print(xx)
#print(shuffle(xx))

a = np.array(range(100))
print a
print a[10:-1]
print a[0:0]

xx = np.array([[0, 1, 2, 3],
               [10, 11, 12, 13],
               [20, 21, 22, 23],
               [30, 31, 32, 33],
               [40, 41, 42, 43]])
print xx
print xx - [1,1,1,1]

print xx - np.mean(xx, axis=0)

print "mean", np.mean(xx, axis=0)


xx = np.random.rand(1000)

normalized, parmsFromNormalization = preprocess1DtoZeroMeanUnit(xx)
new_xx = undoPreprocessing(normalized, parmsFromNormalization)
print "xx", xx
print "mean,std,sqrt", parmsFromNormalization.mean, parmsFromNormalization.std, parmsFromNormalization.sqrtx2
print "newxx", new_xx
print np.allclose(xx, new_xx)


import itertools
for i in itertools.product([1,2,3],['a','b'],[4,5]):
    print i
