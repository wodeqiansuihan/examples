import os
import csv
import ctypes
import matplotlib.pyplot as plt
import numpy as np

def sign(a):
    if a>0:
        return 1;
    elif a<0:
        return -1;
    else:
        return 0;

origin_data = []
fp = open("dd.csv")

for line in fp.readlines():
    line = line.strip("\r")
    line = line.strip("\n")
    line = line.rstrip()
    origin_data.append(float(line))
fp.close()


length = len(origin_data)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(origin_data,'b')


h1= [-0.0757657147892733,-0.0296355276459985,0.497618667632015,0.803738751805916,0.297857795605277,-0.0992195435768472,-0.0126039672620378,0.0322231006040427]
g1= [-0.0322231006040427,-0.0126039672620378,0.0992195435768472,0.297857795605277,-0.803738751805916,0.497618667632015,0.0296355276459985,-0.0757657147892733]

h2 = np.zeros(16)
g2 = np.zeros(16)
h3 = np.zeros(32)
g3 = np.zeros(32)

len_data = length + 1024 

for i in range(8):
    h2[2*i] = 0 
    h2[2*i -1] = h1[i] 
    g2[2*i] = 0 
    g2[2*i-1] = g2[i] 
for i in range(16):
    h3[2*i] = 0 
    h3[2*i -1] = h2[i] 
    g3[2*i] = 0 
    g3[2*i-1] = g2[i] 

data = np.zeros(len_data)
c1 = np.zeros(len_data)
c2 = np.zeros(len_data)
c3 = np.zeros(len_data)


for i in range(512):
    data[i] = origin_data[512 - i] 
    data[length + 512 + i] =origin_data[length - i - 2] 

for i in range(length):
    data[i + 512] = origin_data[i] 


for i in range(len_data):
    temp_c = 0 
    for j in range(8):
        temp_c += h1[j]*data[(i + j)%len_data] 
    c1[i] = temp_c 

for i in range(len_data):
    temp_c = 0 
    for j in range(16):
        temp_c += h2[j]*c1[(i + j)%len_data] 
    c2[i] = temp_c 

for i in range(len_data):
    temp_c = 0 
    for j in range(32):
        temp_c += h3[j]*c2[(i + j)%len_data] 
    c3[i] = temp_c 


for i in range(len_data):
    temp_c = 0 
    for j in range(32):
        temp_c += h3[j] * c3[(i - j + len_data)%(len_data)] 
    c2[i] = temp_c/2 


for i in range(len_data):
    temp_c = 0 
    for j in range(16):
        temp_c += h2[j] * c2[(i - j + len_data)%(len_data)] 
    c1[i] = temp_c/2 


for i in range(len_data):
    temp_c = 0 
    for j in range(8):
        temp_c += h1[j] * c1[(i - j + len_data)%(len_data)] 
    data[i] = temp_c/2 

data = data[512:len_data-512]
length = len(data)
a = np.zeros(length)

for i in range(length - 1):
    a[i] = sign(data[i + 1] - data[i])


ax.plot(data,'r')
plt.show()


