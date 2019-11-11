import matplotlib.pyplot as plt
import numpy as np

orloss = np.array([0.501, 0.506, 0.523, 0.515, 0.520, 
                   0.525, 0.517, 0.513, 0.487, 0.450, 
                   0.425, 0.413, 0.381, 0.332, 0.269, 
                   0.174, 0.099, 0.048, 0.029, 0.020, 
                   0.014, 0.011, 0.012, 0.004, 0.003, 0.005])
p200 = np.array([0.090, 0.095, 0.102, 0.111, 0.121, 
                 0.134, 0.150, 0.168, 0.190, 0.215,
                 0.240, 0.270, 0.305, 0.331, 0.355, 
                 0.370, 0.380, 0.395, 0.410, 0.412, 
                 0.411, 0.403, 0.408, 0.404, 0.399, 0.393])
epoch = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(111)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax1.axis([-0.1, 25.1, 0, 0.55])
ax1.set_ylabel('Orthogonality Loss', font1)

ax2 = ax1.twinx()
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax2.plot(epoch, p200, color='g', linestyle='-', marker='v', linewidth=1.0, label="P@200")
ax2.plot(epoch, orloss, color='b', linestyle='-', marker='^', linewidth=1.0, label="Orthogonality Loss")
ax2.set_ylim([0, 0.55])
ax2.set_ylabel('P@200', font1)
ax1.set_xlabel("Training Epoch", font1)
ax2.legend(loc='upper right', prop=font1)



plt.savefig('./paper/cvpr2020AuthorKit/latex/orth.pdf', format='pdf', pad_inches = 0, bbox_inches='tight')
plt.show()