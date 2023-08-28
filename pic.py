import numpy as np
import matplotlib.pyplot as plt


# plt.xlabel('β')
# plt.ylabel("map")
# xs = np.arange(0, 1.1, 0.1)
# series1 = np.array([0.548, 0.557, 0.564, 0.569, 0.571, 0.579, 0.592, 0.593, 0.593, 0.602, 0.564]).astype(np.double)
# s1mask = np.isfinite(series1)
# series2 = np.array([0.533, 0.536, 0.537, 0.545, 0.548, 0.557, 0.57, 0.568, 0.57, 0.570, 0.531]).astype(np.double)
# s2mask = np.isfinite(series2)
# series3 = np.array([0.5405, 0.5465, 0.5505, 0.557, 0.5595, 0.568, 0.581, 0.5805, 0.5815, 0.586, 0.5475]).astype(np.double)
# s3mask = np.isfinite(series3)
# plt.ylim(0.5, 0.66)
#
# plt.plot(xs[s1mask], series1[s1mask], linestyle='-', label='i2t')
# plt.plot(xs[s2mask], series2[s2mask], linestyle='-', label='t2i')
# plt.plot(xs[s3mask], series3[s3mask], linestyle='-', label='average')
# plt.legend(loc='upper left')
# # 获取当前的坐标轴
# ax = plt.gca()
#
# # 将四个边框设置为不可见
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
#
# plt.savefig('src/parm.png', dpi=72)
# plt.show()

plt.xlabel('λ')
plt.ylabel("map")
xs = np.arange(0, 1.1, 0.1)
series1 = np.array([0.598, 0.6, 0.597, 0.596, 0.595, 0.589, 0.588, 0.591,0.581,0.592,0.583]).astype(np.double)
s1mask = np.isfinite(series1)
series2 = np.array([0.57,0.572,0.566,0.563,0.571,0.56,0.562,0.562,0.56,0.568,0.559]).astype(np.double)
s2mask = np.isfinite(series2)
series3 = np.array([0.584,0.586,0.5815,0.5795,0.583,0.5745,0.575,0.5765,0.5705,0.58,0.571]).astype(np.double)
s3mask = np.isfinite(series3)
plt.ylim(0.54, 0.62)

plt.plot(xs[s1mask], series1[s1mask], linestyle='-', label='i2t')
plt.plot(xs[s2mask], series2[s2mask], linestyle='-', label='t2i')
plt.plot(xs[s3mask], series3[s3mask], linestyle='-', label='average')
plt.legend(loc='upper left')
# 获取当前的坐标轴
ax = plt.gca()

# 将四个边框设置为不可见
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# plt.savefig('src/parm2.png', dpi=72)
plt.show()