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
        #print xdata.shape, loss_data[:,i].shape, val_loss_data[:,i].shape
        plt.plot(xdata, loss_data[:,i], '-', label=headers[i], linewidth=1)
        plt.plot(xdata, val_loss_data[:,i], '--', label='val_'+headers[i], linewidth=1)

    plt.legend(loc=0)
    plt.xlabel('Loss', fontsize=18)
    plt.ylabel('Number of epoches', fontsize=18)

    #plt.yscale('log')
    plt.show()

def draw_dl_scatterplots(data2vis):
    #Multiple figures http://matplotlib.org/examples/pylab_examples/multiple_figs_demo.html
    plt.figure(1, figsize=(20,10))
    plt.subplot(321) #hese are subplot grid parameters encoded as a single integer. For example, "111" means "1x1 grid, first subplot" and "234" means "2x3 grid, 4th subplot".

    #N = 50
    #x = np.random.rand(N)
    #y = np.random.rand(N)
    #colors = np.random.rand(N)
    #area = np.pi * (15 * np.random.rand(N))**2
    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    #plt.scatter(x, y)
    #print type(data2vis['w'].values[0]), data2vis['w'].values

    ierrors = pow(1/(data2vis['rmsep'].values - 0.1),2)
    #print "ierrors", ierrors

    plt.yscale('log')
    plt.xlabel('NN Width', fontsize=18)
    plt.ylabel('lr', fontsize=18)
    plt.scatter(data2vis['w'].values, data2vis['lr'].values, s=ierrors, alpha=0.5)
    #plt.scatter(data2vis['w'], data2vis['lr'], c=data2vis['rmsep'], alpha=0.5)


    print data2vis['rg']
    plt.subplot(322)
    plt.xlabel('Droput', fontsize=18)
    plt.ylabel('L2 Reguarization', fontsize=18)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.scatter(data2vis['dropout'].values, data2vis['rg'].values, s=ierrors, alpha=0.5)

    plt.subplot(323)
    plt.xlabel('Droput', fontsize=18)
    plt.ylabel('Inverted Squared Error', fontsize=18)
    plt.scatter(data2vis['dropout'].values, ierrors, alpha=0.5)

    plt.subplot(324)
    plt.xlabel('NN Width', fontsize=18)
    plt.ylabel('Inverted Squared Error', fontsize=18)
    plt.scatter(data2vis['w'].values, ierrors, alpha=0.5)


    plt.subplot(325)
    plt.xlabel('L2 Reguarization', fontsize=18)
    plt.ylabel('Inverted Squared Error', fontsize=18)
    plt.scatter(data2vis['rg'].values, ierrors, alpha=0.5)

    plt.subplot(326)
    plt.xlabel('Learning Rate', fontsize=18)
    plt.ylabel('Inverted Squared Error', fontsize=18)
    plt.scatter(data2vis['lr'].values, ierrors, alpha=0.5)

    plt.tight_layout()
    plt.show()


#file = open('data/runs.txt', 'r')
file = open('results/may28-50run.txt','r')
data =  file.read()

data = data.replace('\n','')
data = re.sub(r'\[=+\].*?s', '', data)

#print data
#print type(data)

p = re.compile('loss.*?\[(.*?)\].*?val_loss.*?\[(.*?)\].*?NN(.*?)AC_errorRate=([.0-9]+) RMSEP=([.0-9]+)')

p2 = re.compile('([0-9]+)X([0-9]+) dp=([0-9.eE]+)\/([0-9.eE]+).*?\'lr\': (([0-9.eE-]+))')
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
data2vis = []
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

    if float(ac_error) < 29:
    #if float(ac_error) == 40 or float(ac_error) < 33:
    #if float(ac_error) == 40:
        loss_dataset.append(loss_valuesNum)
        val_loss_dataset.append(val_loss_valuesNum)
        columns.append('data'+str(index))
        print "%s = %f, %s, %s, %s " %('data'+str(index), np.min(val_loss_valuesNum), ac_error, rmsep, name)
        index = index+1

    match2 = p2.search(name)
    w = int(match2.group(1))
    d = int(match2.group(2))
    dropout = float(match2.group(3))
    rg = float(match2.group(4))
    lr = float(match2.group(5))
    data2vis.append(np.array([w, d, dropout, rg, lr, float(ac_error), float(rmsep)]))

npdata = np.column_stack(loss_dataset)
val_npdata = np.column_stack(val_loss_dataset)

#print("npdata.shape", npdata.shape)
draw_dl_loss(npdata, val_npdata, columns, 1000)
#npa = np.array(dataset)
#print npa.shape
#df = pd.DataFrame(npa, columns=columns)

data2vis = np.row_stack(data2vis)
data2vis = pd.DataFrame(data2vis, columns=['w', 'd', 'dropout', 'rg', 'lr', 'ac_error', 'rmsep'])
print data2vis.describe()

#draw_dl_scatterplots(data2vis)

#df.plot()
#plt.show()