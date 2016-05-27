import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = open('runs.txt', 'r')
data =  file.read()

print data
print type(data)

p = re.compile('loss.*?\[(.*?)\]')
m = p.search(data )
if m:
    print 'Match found: ', m.group(1)
else:
    print 'No match'


index = 0
dataset = []
columns = []
for match in p.finditer(data):
    values = match.group(1).split(',')
    valuesAsNum = [float(s) for s in values]
    dataset.append(valuesAsNum)
    columns.append('data'+str(index))
    print "%s" % ( valuesAsNum)


npa = np.array(dataset)
print npa.shape
df = pd.DataFrame(npa, columns=columns)

df.plot()
plt.show()