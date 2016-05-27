import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def draw_timeseries_chart(data, headers, maxEntries=50):
    #y_pred_limit = min(maxEntries, len(y_test))
    length = min(maxEntries,data.shape[0])
    xdata = np.array(range(length))
    for i in range(data.shape[1]):
        plt.plot(xdata, data[:,i], label=headers[i], linewidth=1)
        #plt.plot(y_pred_limit, y_pred1, '--', color='r', linewidth=2, label='prediction1')

    plt.legend(loc=0)
    plt.yscale('log')
    plt.show()

file = open('runs.txt', 'r')
data =  file.read()

data = data.replace('\n','')

#print data
#print type(data)

p = re.compile('loss.*?\[(.*?)\].*?DL(.*?)AC_errorRate')
#p = re.compile('loss.*?\[(.*?)\]')
m = p.search(data )
if m:
    print 'G1: ', m.group(1)
    print "G2: ", m.group(2)
else:
    print 'No match'


index = 0
dataset = []
columns = []
for match in p.finditer(data):
    g1 = match.group(1)
    g2 = match.group(2)
    print ">>>g1", g1
    values = g1.split(',')
    valuesAsNum = np.array([float(s) for s in values])
    valuesAsNum.resize((500))
    dataset.append(valuesAsNum)

    print ">>>g2", g2
    name = g2
    print name
    columns.append('data'+str(index))
    index = index+1
    #print "%s %s" % ( valuesAsNum.shape, valuesAsNum)


npdata = np.column_stack(dataset)
#print("npdata.shape", npdata.shape)
draw_timeseries_chart(npdata, columns, 1000)
#npa = np.array(dataset)
#print npa.shape
#df = pd.DataFrame(npa, columns=columns)

#df.plot()
#plt.show()