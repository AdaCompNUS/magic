from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

COLORS = [
    [245 / 255.0, 130 / 255.0, 49 / 255.0], # Orange
    [75 / 255.0, 180 / 255.0, 60 / 255.0], # Green
    [145 / 255.0, 30 / 255.0, 180 / 255.0], # Purple
    [67 / 255.0, 99 / 255.0, 216 / 255.0] # Blue
]

# LightDark
#BASELINES = [64.6, 17.8, -48.3]

# CrowdDrive (on simplified simulator)
#BASELINES = [59.3, 18.7, -5.7]

# PuckPush (on virtual simulator)
BASELINES = [30.0, 50.9, -90.7]

LABELS = [
    'MAGIC',
    'Macro-DESPOT (handcrafted)',
    'DESPOT',
    'POMCPOW'
]

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
        value = tokens[1 + column_index].strip().split(' ')
        value = (float(value[0]), float(value[1][1:-2]))
        data.append((time, value))
        print((time, value))

    print('Min = ', min(data, key=lambda x:x[1][0]))
    print('Max = ', max(data, key=lambda x:x[1][0]))

    fig, axs = plt.subplots(figsize=(17.5, 10.0))
    axs.set_xlabel('Training Iterations', fontsize=52)
    axs.set_ylabel(data_label, fontsize=52)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axs.xaxis.offsetText.set_fontsize(48)
    plt.rc('legend',fontsize=44)
    axs.tick_params(axis='x', labelsize=48)
    axs.tick_params(axis='y', labelsize=48)

    axs.plot(
            [x[0] for x in data],
            [x[1][0] for x in data],
            color=COLORS[0],
            linestyle='solid',
            linewidth=6)
    plt.axhline(y=BASELINES[0], color=COLORS[1], linestyle='--', linewidth=6)
    plt.axhline(y=BASELINES[1], color=COLORS[2], linestyle='--', linewidth=6)
    plt.axhline(y=BASELINES[2], color=COLORS[3], linestyle='--', linewidth=6)
    axs.plot(
            [x[0] for x in data],
            [x[1][0] for x in data],
            color=COLORS[0],
            linestyle='solid',
            linewidth=6)

    plt.legend(LABELS, loc='lower right', framealpha=0.95)



    '''
    axs.fill_between(
            [x[0] for x in data],
            [x[1][0] + 1 * x[1][1] for x in data],
            [x[1][0] - 1 * x[1][1] for x in data],
            color='#050505',
            alpha=0.1,
            interpolate=False)
    '''

    fig.tight_layout()
    # plt.savefig(Path(benchmark_file).parents[0]/'{}.png'.format(data_label))
    plt.show()
