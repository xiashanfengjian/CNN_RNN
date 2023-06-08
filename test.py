import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

f = open('./002358.csv','r')
df = pd.read_csv(f)
num0 = len(df['date'][:])
# print(df['close'][-29:-2])

data = np.zeros((num0-200,201))
for i in range(200,num0):
    for j in range(0,200):
        data[i-200,j] = df['close'][i-199+j]
print(data[:,:])
print(df['close'][200])
# print(df['close'][-1])
df['close'][200:].plot(lw=2)
# plt.show()
for i in range(200,num0-5):
    data[i-200,-1] = 100*(data[i-200+5,-2]-data[i-200,-2])/data[i-200,-2]
print(data[:,:])
np.savetxt('./test.txt',data,fmt='%.02f')