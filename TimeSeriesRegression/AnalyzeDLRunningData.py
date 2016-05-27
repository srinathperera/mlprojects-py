import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def draw_timeseries_chart(data, headers, maxEntries=50):
    #y_pred_limit = min(maxEntries, len(y_test))
    length = min(maxEntries,data.shape[0])
    xdata = np.array(range(length))
    for i in range(data.shape[1]):
        plt.plot(xdata, data[:,i], label=headers[i], linewidth=1) #see http://matplotlib.org/api/pyplot_api.html for other type of lines

        #plt.plot(y_pred_limit, y_pred1, '--', color='r', linewidth=2, label='prediction1')

    plt.legend(loc=0)
    plt.yscale('log')
    plt.show()

def draw_dl_loss(loss_data, val_loss_data, headers, maxEntries=50):
    #y_pred_limit = min(maxEntries, len(y_test))
    length = min(maxEntries,loss_data.shape[0])
    xdata = np.array(range(length))
    for i in range(loss_data.shape[1]):
        print xdata.shape, loss_data[:,i].shape, val_loss_data[:,i].shape
        plt.plot(xdata, loss_data[:,i], '+', label=headers[i], linewidth=1)
        plt.plot(xdata, val_loss_data[:,i], '--', label='val_'+headers[i], linewidth=1)

    plt.legend(loc=0)
    plt.yscale('log')
    plt.show()

file = open('data/runs.txt', 'r')
data =  file.read()

data = data.replace('\n','')
data = re.sub(r'\[=+\].*?s', '', data)

#print data
#print type(data)

p = re.compile('loss.*?\[(.*?)\].*?val_loss.*?\[(.*?)\].*?NN(.*?)AC_errorRate=([.0-9]+) RMSEP=([.0-9]+)')
#p = re.compile('loss.*?\[(.*?)\]')
#m = p.search(data )
#if m:
#    print 'G1: ', m.group(1)
#    print "G2: ", m.group(2)
#else:
#    print 'No match'


index = 0
loss_dataset = []
val_loss_dataset = []

columns = []
for match in p.finditer(data):
    loss_str = match.group(1)
    val_loss_str = match.group(2)
    name = match.group(3)
    ac_error = match.group(4)
    rmsep = match.group(5)
    #print ">>>g1", g1
    loss_values = loss_str.split(',')
    loss_valuesNum = np.array([float(s) for s in loss_values])
    loss_valuesNum.resize((500))

    val_loss_values = val_loss_str.split(',')
    val_loss_valuesNum = np.array([float(s) for s in val_loss_values])
    val_loss_valuesNum.resize((500))

    #if float(ac_error) < 35:
    #if float(ac_error) == 40 or float(ac_error) < 33:
    if float(ac_error) == 40:
        loss_dataset.append(loss_valuesNum)
        val_loss_dataset.append(val_loss_valuesNum)
        columns.append('data'+str(index))
        print "%s = %f, %s, %s, %s " %('data'+str(index), np.min(val_loss_valuesNum), ac_error, rmsep, name)
        index = index+1

npdata = np.column_stack(loss_dataset)
val_npdata = np.column_stack(val_loss_dataset)

#print("npdata.shape", npdata.shape)
draw_dl_loss(npdata, val_npdata, columns, 1000)
#npa = np.array(dataset)
#print npa.shape
#df = pd.DataFrame(npa, columns=columns)

#df.plot()
#plt.show()