import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

t = np.linspace(0, np.pi/2, 100)

# true pos value
x1 = np.hstack([np.linspace(-1.0, 0.0, 100), 2.0 * np.sin(t)]) + 1.0
y1 = np.hstack([[1.0]*100, 1.0 * np.cos(t)]) - 1.0
x2 = np.hstack([[0.0]*100, 1.0 * np.sin(t)]) + 1.0
y2 = np.hstack([[2.0]*100, 2.0 * np.cos(t)]) - 1.0

# Observed  value
v1x = np.diff(x1)
v1y = np.diff(y1)
v1 = np.sqrt(v1x**2 + v1y**2)
theta = np.hstack([[0], np.arctan(np.diff(y1)/np.diff(x1))])
theta[np.isnan(theta)] = 0.0
X2 = (x2-x1)*np.cos(theta) + (y2-y1)*np.sin(theta)
Y2 = (y2-y1)*np.cos(theta) - (x2-x1)*np.sin(theta)

# Estimated value
x1e = []
y1e = []
for i,v in enumerate(v1):
    if i == 0:
        x1e.append(0)
        y1e.append(0)
    else:
        x1e.append(x1e[i-1] + v*np.cos(theta[i]))
        y1e.append(y1e[i-1] + v*np.sin(theta[i]))
x1e.append(x1e[-1])
y1e.append(y1e[-1])
x2e = X2*np.cos(theta) - Y2*np.sin(theta) + x1e
y2e = X2*np.sin(theta) + Y2*np.cos(theta) + y1e

def update(val):
    _idx = min(int(slider.val), len(v1)-1)
    ax3.clear()
    ax3.plot(np.arange(0,len(v1))*0.1, v1, 'k')
    ax3.plot(_idx * 0.1, v1[_idx], 'ko')
    fig.canvas.draw_idle()

fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,2), (0,0))
ax2 = plt.subplot2grid((2,2), (0,1))
ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
axamp = plt.axes([0.25, .03, 0.50, 0.02])                                                                                                            
slider = Slider(axamp, 'linepos', 0, len(x1), valinit=0, valfmt='%d')
slider.on_changed(update)

ax1.plot(x1, y1, linewidth=2, color='k')
ax1.plot(x2, y2, linewidth=2, color='b')
ax1.plot(x1e, y1e, linewidth=2, color='r')
ax1.plot(x2e, y2e, linewidth=2, color='g')
ax1.set_aspect('equal')
ax1.grid(True)

ax2.plot(X2, Y2, linewidth=2, color='k')
ax2.set_aspect('equal')
ax2.grid(True)


plt.show()
