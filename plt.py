import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,50)
y = x ** 2 - x

fig = plt.figure()
ax = fig.add_subplot(2,2,4)
ax.plot(x,y,label='ding')
ax.set_xlabel('time/s')
ax.set_ylabel('amp')
ax.set_title('test')
ax.legend()

plt.show()

