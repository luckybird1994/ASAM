import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x_values = np.arange(-0.2, 1.3, 0.2)
y_values = np.clip(x_values, 0, 1)

# 绘制折线图
plt.plot(x_values, y_values,  linestyle='-', color='red')

# 设置坐标轴范围
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

# 设置坐标轴标签
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# 设置坐标刻度
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))

# 显示图例
# plt.legend(['y = x'])

# 显示图形
plt.savefig('dbp.png')
plt.show()