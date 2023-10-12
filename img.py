import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(6, 2))
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
# 生成50个随机数作为y值
data = torch.tensor([[0.0686],
        [0.1113],
        [0.1699],
        [0.1337],
        [0.1360],
        [0.1338],
        [0.1039],
        [0.0740],
        [0.0687]
        ], )
# 生成x值
x = np.arange(len(data))
x_new = np.linspace(0, len(data) - 1, 300)
spl = make_interp_spline(np.arange(len(data)), data, k=3)
y_new = spl(x_new)

# 绘制曲线

plt.plot(x_new, y_new,  color="blue",label='weight')

plt.ylim(0, 0.25)
plt.xlim(-1, 9)

plt.xticks([])

# 添加标题和标签

plt.legend()
# 显示图表
plt.show()
