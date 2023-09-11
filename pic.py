import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False, #解决负号无法显示的问题
    "font.size": '12'
}
rcParams.update(config)

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

plt.figure(figsize=(8,4.5))
plt.xlabel('λ', fontsize=12)
plt.ylabel("mAP", fontsize=12)
xs = np.arange(0, 1.1, 0.1)
series1 = np.array([0.597, 0.601, 0.599, 0.596, 0.593, 0.589, 0.586, 0.584, 0.592, 0.593, 0.587]).astype(np.double)
s1mask = np.isfinite(series1)
series2 = np.array([0.572, 0.578, 0.571, 0.571, 0.569, 0.562, 0.563, 0.566, 0.564, 0.568, 0.561]).astype(np.double)
s2mask = np.isfinite(series2)
series3 = np.array([0.5845, 0.5895, 0.585, 0.5835, 0.581, 0.5755, 0.5745, 0.575, 0.578, 0.5805, 0.574]).astype(np.double)
s3mask = np.isfinite(series3)
plt.ylim(0.55, 0.62)

plt.plot(xs[s1mask], series1[s1mask], linestyle='-', label='I2T')
plt.plot(xs[s2mask], series2[s2mask], linestyle='-', label='T2I')
plt.plot(xs[s3mask], series3[s3mask], linestyle='-', label='average')
plt.legend(loc='upper right')
# 获取当前的坐标轴
ax = plt.gca()
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)
# plt.yaxis.set_minor_locator(minor_locator)
plt.grid(which='both', linewidth=0.5)

# 将四个边框设置为不可见
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('src/parm2.png', dpi=72)
plt.savefig('src/parm2.pdf', dpi=72)
plt.show()

plt.figure(figsize=(8,4.5))
plt.xlabel('epoch', fontsize=12)
plt.ylabel("mAP", fontsize=12)
xs = np.arange(0, 101, 1)
i2t = np.array([0.1277, 0.3969, 0.4364, 0.4682, 0.4944, 0.492, 0.5075, 0.5209, 0.5334, 0.5385, 0.5418, 0.5503, 0.5593, 0.5621, 0.5602, 0.5679, 0.5688, 0.567, 0.5718, 0.5678, 0.5742, 0.5721, 0.5673, 0.5692, 0.5682, 0.5645, 0.5741, 0.5744, 0.5755, 0.5729, 0.5638, 0.5698, 0.568, 0.5652, 0.5656, 0.5629, 0.565, 0.557, 0.5555, 0.5486, 0.5523, 0.5528, 0.5444, 0.5491, 0.5374, 0.5386, 0.534, 0.5275, 0.5327, 0.52, 0.5296, 0.5167, 0.515, 0.5133, 0.5102, 0.5076, 0.5088, 0.5069, 0.4933, 0.4976, 0.4966, 0.4953, 0.486, 0.489, 0.4882, 0.4802, 0.4749, 0.475, 0.4735, 0.4782, 0.4658, 0.4608, 0.4542, 0.4583, 0.451, 0.4476, 0.4484, 0.4492, 0.445, 0.4389, 0.4453, 0.4429, 0.4428, 0.4374, 0.4334, 0.4346, 0.4323, 0.4315, 0.4286, 0.4292, 0.4309, 0.4301, 0.4277, 0.4284, 0.4279, 0.4277, 0.4276, 0.4274, 0.4273, 0.4271, 0.4271]).astype(np.double)
s1mask = np.isfinite(i2t)
t2i = np.array([0.1266, 0.3767, 0.435, 0.4567, 0.4756, 0.4868, 0.4915, 0.5054, 0.5111, 0.518, 0.5306, 0.5331, 0.5386, 0.5414, 0.543, 0.542, 0.5473, 0.5511, 0.5522, 0.5503, 0.5501, 0.5476, 0.5499, 0.5521, 0.5472, 0.5477, 0.5481, 0.5496, 0.5455, 0.5496, 0.5415, 0.5383, 0.5394, 0.538, 0.5379, 0.5343, 0.5318, 0.5311, 0.5283, 0.527, 0.5233, 0.5262, 0.5171, 0.5212, 0.5077, 0.5124, 0.5091, 0.5009, 0.5047, 0.4948, 0.5006, 0.4941, 0.4926, 0.4934, 0.4868, 0.4861, 0.4854, 0.4831, 0.4759, 0.4753, 0.4727, 0.4695, 0.464, 0.4642, 0.4664, 0.4621, 0.4563, 0.4536, 0.453, 0.4558, 0.4475, 0.445, 0.4359, 0.4399, 0.4335, 0.4314, 0.4317, 0.4323, 0.4281, 0.4245, 0.428, 0.4272, 0.4266, 0.4215, 0.4191, 0.4188, 0.416, 0.4159, 0.4123, 0.4131, 0.4139, 0.4145, 0.4122, 0.4124, 0.4116, 0.4111, 0.411, 0.4109, 0.4109, 0.4107, 0.4107]).astype(np.double)
s2mask = np.isfinite(t2i)
i2t2 = np.array([0.1223, 0.4509, 0.5265, 0.5737, 0.5847, 0.589, 0.5769, 0.5861, 0.5944, 0.5943, 0.5905, 0.5932, 0.5902, 0.5927, 0.5912, 0.5918, 0.5923, 0.5926, 0.5891, 0.5942, 0.5957, 0.5961, 0.5976, 0.5952, 0.5964, 0.5958, 0.595, 0.5957, 0.595, 0.5981, 0.5946, 0.5973, 0.598, 0.5994, 0.5971, 0.5969, 0.5944, 0.5963, 0.5925, 0.5939, 0.5963, 0.5936, 0.5965, 0.5956, 0.5955, 0.5962, 0.5941, 0.5949, 0.5931, 0.5934, 0.5921, 0.5944, 0.5938, 0.5912, 0.595, 0.5927, 0.5921, 0.5909, 0.5936, 0.5937, 0.5916, 0.5899, 0.5891, 0.5896, 0.5912, 0.5905, 0.5894, 0.5894, 0.5896, 0.5903, 0.5879, 0.5875, 0.5867, 0.5848, 0.5852, 0.587, 0.5857, 0.5851, 0.5828, 0.5847, 0.5851, 0.586, 0.5839, 0.5851, 0.5848, 0.5842, 0.585, 0.5853, 0.5852, 0.585, 0.5851, 0.5851, 0.5849, 0.5849, 0.5851, 0.5849, 0.5849, 0.5848, 0.5848, 0.5848, 0.5847]).astype(np.double)
s3mask = np.isfinite(i2t2)
t2i2 = np.array([0.1364, 0.4542, 0.4888, 0.558, 0.56, 0.5682, 0.5602, 0.5709, 0.5688, 0.5668, 0.5707, 0.5675, 0.5729, 0.5732, 0.5682, 0.5684, 0.5734, 0.5693, 0.5687, 0.5778, 0.5752, 0.5731, 0.5743, 0.5754, 0.5751, 0.5726, 0.5709, 0.5719, 0.5708, 0.5763, 0.5683, 0.5684, 0.57, 0.5705, 0.5712, 0.5707, 0.5698, 0.5681, 0.5692, 0.5688, 0.5718, 0.5687, 0.5709, 0.5727, 0.5698, 0.5694, 0.5696, 0.5698, 0.5723, 0.5669, 0.5673, 0.5686, 0.5676, 0.5651, 0.569, 0.5668, 0.5682, 0.566, 0.569, 0.5684, 0.5665, 0.5658, 0.5641, 0.5659, 0.5663, 0.5644, 0.5641, 0.5651, 0.5646, 0.5646, 0.5642, 0.5643, 0.562, 0.56, 0.562, 0.5624, 0.5614, 0.5607, 0.5601, 0.5592, 0.5601, 0.5617, 0.5592, 0.5609, 0.5602, 0.5603, 0.5595, 0.5593, 0.5595, 0.5592, 0.5593, 0.5591, 0.5591, 0.5589, 0.559, 0.5589, 0.5588, 0.5588, 0.5589, 0.5588, 0.5588]).astype(np.double)
s4mask = np.isfinite(t2i2)
plt.ylim(0.1, 0.63)
minor_locator = AutoMinorLocator(2)
ax = plt.gca()
ax.xaxis.set_minor_locator(minor_locator)
# plt.yaxis.set_minor_locator(minor_locator)
plt.grid(which='both', linewidth=0.5)

plt.plot(xs[s1mask], i2t[s1mask], linestyle='-', color="cornflowerblue", label='MRL-I2T')
plt.plot(xs[s2mask], t2i[s2mask], linestyle='dashed', color="cornflowerblue", label='MRL-T2I')
plt.plot(xs[s3mask], i2t2[s3mask], linestyle='-', color="firebrick", label='DMCM-I2T')
plt.plot(xs[s4mask], t2i2[s4mask], linestyle='dashed', color="firebrick", label='DMCM-T2I')
plt.legend(loc='lower right')
# 获取当前的坐标轴
ax = plt.gca()

# 将四个边框设置为不可见
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('src/result.png', dpi=72)
plt.savefig('src/result.pdf', dpi=72)
plt.show()

