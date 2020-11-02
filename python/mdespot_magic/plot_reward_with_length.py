from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    data_file = sys.argv[1]

    # Extract column.
    with open(data_file, 'r') as f:
        raw_lines = f.readlines()
    data = []
    for l in raw_lines:
        tokens = l.split('|')
        length = int(tokens[0])
        value = tokens[1].strip().split(' ')
        value = (float(value[0]), float(value[1][1:-2]))
        data.append((length, value))

    fig, axs = plt.subplots(figsize=(17.5, 8))
    axs.set_xlabel('Macro-action Length', fontsize=52)
    axs.set_ylabel('Acc. Reward', fontsize=52)
    plt.rc('legend',fontsize=26)
    axs.tick_params(axis='x', labelsize=32)
    axs.tick_params(axis='y', labelsize=32)
    plt.xticks(sorted(x[0] for x in data))

    axs.plot(
            [x[0] for x in data],
            [x[1][0] for x in data],
            color='blue',
            linestyle='solid',
            linewidth=6)
    plt.errorbar([x[0] for x in data], [x[1][0] for x in data], yerr=[x[1][1] for x in data],
            ecolor='black', mfc='black', mec='black',
            fmt='o', capsize=5, elinewidth=3, markeredgewidth=3)


    fig.tight_layout()
    # plt.savefig(Path(benchmark_file).parents[0]/'{}.png'.format(data_label))
    plt.show()
