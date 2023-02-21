from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt


#归一化
data = np.random.random_integers(0, 100, (100, 2))
mm = MinMaxScaler()
mm_data = mm.fit_transform(data)
origin_data = mm.inverse_transform(mm_data)
print('data is ',data)
print('after Min Max ',mm_data)
print('origin data is ',origin_data)

gs = gridspec.GridSpec(5,5)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0:2, 1:4])
ax2 = fig.add_subplot(gs[3:5, 1:4])
 
ax1.scatter(data[:, 0], data[:, 1])
ax2.scatter(mm_data[:, 0], mm_data[:, 1])
 
plt.show()

#标准化
cps = np.random.random_integers(0, 100, (100, 2))
 
ss = StandardScaler()
std_cps = ss.fit_transform(cps)
 
gs = gridspec.GridSpec(5,5)
fig = plt.figure()
ax1 = fig.add_subplot(gs[0:2, 1:4])
ax2 = fig.add_subplot(gs[3:5, 1:4])
 
ax1.scatter(cps[:, 0], cps[:, 1])
ax2.scatter(std_cps[:, 0], std_cps[:, 1])
 
plt.show()