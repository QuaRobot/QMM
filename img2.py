import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6, 2))

waters = ('A', 'B', 'C', 'D','e','f4' )
#buy_number = [0.2607,0.1774,0.2728,0.1209,0.1683]
buy_number = [0.1864,0.1185,0.1756,0.1469,0.2134,0.1592]

plt.barh(waters, buy_number,height=0.3,color='blue') # 横放条形图函数 barh
plt.xlabel('weight')
plt.show()
