import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['1-Layer', '2-Layer', '3-Layer']
MLP_model = [0.8979, 0.8877, 0.891]
Conv_Net_Model = [0.9108, 0.9248, 0.9263]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, MLP_model, width, label='MLP Model')
rects2 = ax.bar(x + width/2, Conv_Net_Model, width, label='ConvNet Model')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (in %)')
ax.set_title('Accuracy on Test Data for MLP & ConvNet Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
