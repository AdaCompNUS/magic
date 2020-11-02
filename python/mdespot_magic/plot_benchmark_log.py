from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    benchmark_file = sys.argv[1]
    column_index = int(sys.argv[2])
    data_label = sys.argv[3]

    # Extract column.
    with open(benchmark_file, 'r') as f:
        raw_lines = f.readlines()
    data = []
    for l in raw_lines:
        tokens = l.split('|')
        time = int(tokens[0])
        if time > 0:
            value = tokens[1 + column_index].strip().split(' ')
            value = (float(value[0]), float(value[1][1:-2]))
            data.append((time, value))

    print('Min = ', min(data, key=lambda x:x[1][0]))
    print('Max = ', max(data, key=lambda x:x[1][0]))

    fig, axs = plt.subplots(figsize=(17.5, 8))
    axs.set_xlabel('Training Iterations', fontsize=52)
    axs.set_ylabel(data_label, fontsize=52)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axs.xaxis.offsetText.set_fontsize(24)
    plt.rc('legend',fontsize=26)
    axs.tick_params(axis='x', labelsize=32)
    axs.tick_params(axis='y', labelsize=32)

    axs.plot(
            [x[0] for x in data],
            [x[1][0] for x in data],
            color='black',
            linestyle='solid',
            linewidth=6)
    axs.fill_between(
            [x[0] for x in data],
            [x[1][0] + 1 * x[1][1] for x in data],
            [x[1][0] - 1 * x[1][1] for x in data],
            color='#050505',
            alpha=0.1,
            interpolate=False)

    fig.tight_layout()
    # plt.savefig(Path(benchmark_file).parents[0]/'{}.png'.format(data_label))
    plt.show()
