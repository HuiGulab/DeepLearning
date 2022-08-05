# %matplotlib inline
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


# 再次使用numpy进行可视化
x = np.linspace(-7, 7, 50)


# Figure 并指定大小 图表大小
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, normal(x, 0, 1), color='blue', linewidth=1.0, linestyle='-')
plt.plot(x, normal(x, 0, 2), color='red', linewidth=1.0, linestyle='-')
plt.plot(x, normal(x, 3, 1), color='green', linewidth=1.0, linestyle='-')

# 设置 x，y 轴的范围以及 label 标注
plt.xlim(0, 3)
plt.ylim(0, 0.5)
plt.xlabel('x')
plt.ylabel('y')

# 设置坐标轴刻度线
# Tick X 范围 (-1，2) Tick Label(-1，-0.25，0.5，1.25，2) 刻度数量 5 个
# new_ticks = np.linspace(-1, 2, 5)
new_ticks = np.linspace(-6, 6, 7)
plt.xticks(new_ticks)

# Tick Y 范围(-2.2,-1,1,1.5,2.4) ，Tick Label (-2.2, -1, 1, 1.5, 2.4) 别名(下面的英文)
# plt.yticks([-2.2, -1, 1, 1.5, 2.4],
#            [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
new_ticks = np.linspace(0, 0.4, 5)
plt.yticks(new_ticks)

# 设置坐标轴 gca() 获取坐标轴信息
ax = plt.gca()
# 使用.spines设置边框：x轴；将右边颜色设置为 none。
# 使用.set_position设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 移动坐标轴
# 将 bottom 即是 x 坐标轴设置到 y=0 的位置。
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))

# 将 left 即是 y 坐标轴设置到 x=0 的位置。
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# 设置标签
# ax.set_title('y = x^2', fontsize=14, color='r')

# 显示图像
plt.show()
