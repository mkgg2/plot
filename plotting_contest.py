import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as sp
import random
import math


def logi(x, t):
    return 1 / (1 + np.exp(-x + t))

def revlogi(x, t):
    return 1 - (1 / (1 + np.exp(-x + t)))

def wall(x, r):
    lx9 = logi(x, 270)
    lr9 = logi(r, 0)
    lx1 = revlogi(x, 30)
    lr1 = revlogi(r, 0)
    result = lx9 * lr9 + lx1 * lr1
    return 1 - result

class ameba:
    def __init__(self):
        self.frame_count = 0
        self.dfx = []
        self.dfy = []
        self.a = 150
        self.b = 150

    def rand_norm(self, x, y, sdx, sdy, n):
        for i in range(n):
            self.dfx.append(random.normalvariate(x, sdx))
            self.dfy.append(random.normalvariate(y, sdy))
            self.frame_count += 1
            self.a = x
            self.b = y

    def rand_uni(self, xmin, xmax, ymin, ymax, n):
        for i in range(n):
            self.dfx.append(random.uniform(xmin, xmax))
            self.dfy.append(random.uniform(ymin, ymax))
            self.frame_count += 1

    def rand_uni_norm(self, xmin, xmax, y, ysd, n):
        for i in range(n):
            self.dfx.append(random.uniform(xmin, xmax))
            self.dfy.append(random.normalvariate(y, ysd))
            self.frame_count += 1

    def rand_norm_uni(self, x, xsd, ymin, ymax, n):
        for i in range(n):
            self.dfx.append(random.normalvariate(x, xsd))
            self.dfy.append(random.uniform(ymin, ymax))
            self.frame_count += 1

    def liss(self, radmin, radmax, x, y, n):
        for i in range(n):
            self.th = random.uniform(radmin, radmax)
            self.dfx.append(150 + 150 * np.sin(x * math.radians(self.th)))
            self.dfy.append(150 + 150 * np.sin(y * math.radians(self.th)))
            self.frame_count += 1

    def circle(self, radmin, radmax, n):
        for i in range(n):
            self.th = random.uniform(radmin, radmax)
            self.dfx.append(150 + 150 * np.sin(math.radians(self.th)))
            self.dfy.append(150 + 150 * np.cos(math.radians(self.th)))
            self.frame_count += 1

    def circle2(self, radmin, radmax, nn, n):
        for i in range(n):
            self.th = random.uniform(radmin, radmax)
            self.dfx.append(150 + (150 - nn) * np.sin(math.radians(self.th)))
            self.dfy.append(150 + (150 - nn) * np.cos(math.radians(self.th)))
            self.frame_count += 1

    def move_random(self, n, t):
        for i in range(n):
            self.rand_norm(self.a, self.b, 15, 15, t)
            self.randx = random.uniform(-20, 20)
            self.randy = random.uniform(-20, 20)
            self.randxx = wall(self.a,  self.randx) * self.randx
            self.randyy = wall(self.b,self.randy) * self.randy
            self.a += self.randxx
            self.b += self.randyy

    def dataframe(self):
        self.df = pd.DataFrame({'x': self.dfx, 'y': self.dfy})

ins = ameba()
#-------------------------------------------------------------------------
ins.rand_norm(30, 30, 15, 15, 3000)
for j in range(5):
    ins.rand_norm(30 + j * 30, 30 + j * 30, 15, 15, 1000)
ins.rand_norm(150, 150, 15, 15, 1000)
for j in range(4):
    ins.rand_norm(ins.a, ins.b + 30, 15, 15, 500)
    ins.rand_norm(ins.a + 30, ins.b, 15, 15, 500)
ins.rand_norm(270, 270, 15, 15, 1000)
ins.rand_norm(150, 150, 15, 15, 3000)
ins.move_random(300, 100)
ins.rand_uni(0, 60, 0, 300, 3000)
ins.rand_uni(0, 60, 0, 60, 1000)
ins.rand_uni(0, 300, 0, 60, 3000)
ins.rand_uni(240, 300, 0, 60, 1000)
ins.rand_uni(240, 300, 0, 300, 3000)
ins.rand_uni(240, 300, 240, 300, 1000)
ins.rand_uni(0, 300, 240, 300, 3000)
ins.rand_norm(150, 300, 15, 15, 1000)
for i in range(72):
    ins.circle(i * 10, (i + 1) * 10, 100)
for i in range(72):
    ins.circle2(i * 10, 0, 150 * (i / 71), 100)
for i in range(72):
    ins.liss(i * 5, (i + 1) * 5, 2, 4, 100)
for i in range(72):
    ins.liss(i * 5, (i + 1) * 5, 3, 4, 100)
ins.rand_norm(150, 150, 15, 15, 1000)
for j in range(5):
    ins.rand_norm(150 - j * 30, 150 - j * 30, 15, 15, 500)
ins.rand_norm(30, 30, 15, 15, 3000)
#-------------------------------------------------------------------------
ins.dataframe()

s = 0
total_frame = (ins.frame_count / 1000 - 1) * 1000
fig = plt.figure(figsize=(9, 10))
plt.style.use('ggplot')

grid = fig.add_gridspec(8, 6, wspace=0, hspace=0)
ax_top = fig.add_subplot(grid[0, 0:5])
ax_right = fig.add_subplot(grid[1:6, 5])
ax_bottom_left = fig.add_subplot(grid[-1, 0:5])
ax_bottom_right = fig.add_subplot(grid[7, 5])
ax_main = fig.add_subplot(grid[1:6, 0:5])

xgrid = np.linspace(0, 300, 100)
ygrid = np.linspace(0, 300, 100)
xgrid1d = np.linspace(0, 300, 3000)

def kde1d(x):
    kde = sp.gaussian_kde(x)
    Z = kde(xgrid1d)
    return Z

dfxmean = [30] * 100
dfymean = [30] * 100

def plot(data):
    global ax_top, ax_right, ax_main, ax_bottom_left, ax_bottom_right, s, dfxmean, dfymean
    ax_top.clear()
    ax_right.clear()
    ax_main.clear()
    ax_bottom_left.clear()
    ax_bottom_right.clear()

    ax_top.set_xlim(0, 300)
    ax_right.set_ylim(0, 300)
    ax_main.set_xlim(0, 300)
    ax_main.set_ylim(0, 300)
    ax_bottom_left.set_xlim(0, 100)
    ax_bottom_left.set_ylim(0, 350)
    ax_bottom_right.set_ylim(0, 350)
    ax_bottom_right.set_xlim(0, 100)

    ax_top.axis("off")
    ax_right.axis("off")
    ax_bottom_left.axis("off")
    ax_bottom_right.axis("off")

    x1 = kde1d(ins.df[s:s + 1000].x)
    y1 = kde1d(ins.df[s:s + 1000].y)

    dfxmean.pop(0)
    dfymean.pop(0)
    dfxmean.append(np.mean(ins.df[s:s + 1000].x))
    dfymean.append(np.mean(ins.df[s:s + 1000].y))

    ax_top.scatter(xgrid1d, x1, 0.1, c=x1, cmap="Reds")
    ax_right.scatter(y1, xgrid1d, 0.1, c=y1, cmap="Reds")
    ax_main = sns.kdeplot(ins.df[s:s + 1000].x, ins.df[s:s + 1000].y, shade=True, shade_lowest=False, cmap="Blues")
    ax_bottom_left.plot(range(100), dfxmean, color="Blue", alpha=0.5, label="x")
    ax_bottom_left.plot(range(100), dfymean, color="Red", alpha=0.5, label="y")
    ax_bottom_right.annotate('', xytext=[100, 150], xy=[0, np.mean(ins.df[s:s + 1000].x)],
                             arrowprops=dict(arrowstyle='wedge',
                                             connectionstyle='arc3',
                                             facecolor='blue',
                                             edgecolor='blue')
                             )
    ax_bottom_right.annotate('', xytext=[100, 150], xy=[0, np.mean(ins.df[s:s + 1000].y)],
                             arrowprops=dict(arrowstyle='wedge',
                                             connectionstyle='arc3',
                                             facecolor='red',
                                             edgecolor='red')
                             )
    ax_bottom_left.legend(loc="upper left")

    plt.tight_layout()

    print("\b" * 200, s,
          " / ",
          total_frame,
          end="")
    s += 100


ani_frame_num = int(ins.frame_count / 1000 - 1) * 10

ani = animation.FuncAnimation(fig, plot, interval=100, frames=ani_frame_num)
ani.save("plot.mp4", writer="ffmpeg", dpi=200)
