from collections import defaultdict
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.ticker as tkr


def load(filename):
    result = []
    labels = []
    smallest = float('inf')
    largest = 0
    with open(filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            values = [float(x) for x in line[1:]]
            result.append(values)
            labels.append(name_of_matrix)
            for i in values:
                largest = max(i, largest)
                smallest = min(i, smallest)
    return np.asarray(result), labels
  
def get_s_l(filename):
    other_filename = filename.replace('encoder', 'decoder')
    if 'l1_decoder' in filename:
        other_filename = filename.replace('l1_decoder', 'l1_encoder')
    if 'cossim_decoder' in filename:
        other_filename = filename.replace('cossim_decoder', 'cossim_encoder')
    result = []
    labels = []
    smallest = float('inf')
    largest = 0
    with open(filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            values = [float(x) for x in line[1:]]
            result.append(values)
            labels.append(name_of_matrix)
            for i in values:
                largest = max(i, largest)
                smallest = min(i, smallest)

    with open(other_filename) as file:
        for line in file:
            line = line.strip().split('\t')
            name_of_matrix = line[0]
            values = [float(x) for x in line[1:]]
            result.append(values)
            labels.append(name_of_matrix)
                for i in values:
                    largest = max(i, largest)
                    smallest = min(i, smallest)
    return smallest, largest

filenames = ["l1_decoder_t5.tsv", "l1_encoder_t5.tsv"]

for j, name in enumerate(names):
    matrix, labels = load(name)
    s, l = get_s_l(name)
    j = j // 2
    color = "Blues" if "l1_encoder" in name or "l1_decoder" in name else "Greens"
    new_labels = []
    for item in labels:
        if item == "wo":
            item = "w_o"
        if item == "wi":
            item = "w_i"
        if "x" in item:
            item = item[-1] + "_x"
        item = "$" + item + "$"
        new_labels.append(item)
    cbar_val = True
    formatter = tkr.ScalarFormatter()
    formatter.set_powerlimits((0, 0))
    ax = sns.heatmap(matrix, cmap=color, vmax=l, vmin=s, yticklabels=new_labels, xticklabels=range(24), square=True, cbar=cbar_val, cbar_kws={"shrink": .35, "format": formatter})
    cbar_axes = ax.figure.axes[-1]
    cbar_axes.yaxis.label.set_size(2)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    fig = ax.get_figure()
    for i in range(matrix.shape[0] + 1):
        ax.axhline(i, color='white', lw=2)
    for i in range(matrix.shape[1] + 1):
        ax.axvline(i, color='white', lw=2)
    fig.tight_layout()
    fig.savefig(f"images/{name.split('.')[0]}.png", dpi=300, bbox_inches='tight', pad_inches = 0)
    fig.clf()