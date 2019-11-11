import matplotlib.pyplot as plt
import numpy as np

sk_a = np.array([0.361, 0.380, 0.390, 0.399, 0.407, 0.412, 0.411, 0.409, 0.399, 0.389, 0.372])
sk_u = np.array([0.369, 0.379, 0.389, 0.397, 0.395, 0.392, 0.387, 0.380, 0.372, 0.366, 0.351])
tu = np.array([0.152, 0.159, 0.166, 0.172, 0.176, 0.173, 0.169, 0.164, 0.160, 0.154, 0.147])

x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
font1 = {'family' : 'Times New Roman',
'size'   : 9,
}

figure, ax = plt.subplots(figsize=(8,4))
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.axis([-0.1, 1.1, 0, 0.54])
plt.xlabel("$\omega$", font1)
plt.ylabel("P@200", font1)
plt.plot(x, sk_a, color='r', linestyle='-', marker='v', linewidth=1.0, label="Sketchy Ext. (aligned)")
plt.plot(x, sk_u, color='b', linestyle='-', marker='^', linewidth=1.0, label="Sketchy Ext. (unaligned)")
plt.plot(x, tu, color='c', linestyle='-', marker='d', linewidth=1.0, label="TU-Berlin Ext.")
plt.legend(loc='upper right', prop=font1)

plt.savefig('./paper/cvpr2020AuthorKit/latex/omega_fig.pdf', format='pdf', pad_inches = 0, bbox_inches='tight')
plt.show()