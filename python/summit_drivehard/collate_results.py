import numpy as np
import argparse
import os

def stat_moments(m1, m2, n):
    return (m1 / n, np.sqrt(((m2 / n - (m1 / n)**2)) * float(n) / float(n - 1)) / np.sqrt(n))

def stat_list(l):
    return (np.mean(l), np.std(l, ddof=1) / np.sqrt(len(l)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    data = []

    folder = args.folder
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r') as f:
            data.extend([[float(x) for x in l.strip('\n').split()] for l in f.readlines()])

    data = np.array(data)
    episode_count = len(data)
    step_count = sum(data[:,12])

    print('Episode count = {}'.format(episode_count))
    print('Step count = {}'.format(step_count))
    print('Reward = {}'.format(stat_list(data[:,0])))
    print('Progress = {}'.format(stat_list(data[:,1])))
    print('Speed = {}'.format(stat_moments(
        sum(data[:,2]), sum(data[:,3]), step_count)))
    print('Steer = {}'.format(stat_moments(
        sum(data[:,4]), sum(data[:,5]), step_count)))
    print('Steer factor = {}'.format(stat_moments(
        sum(data[:,6]), sum(data[:,7]), step_count)))
    print('Stall = {}'.format(stat_moments(
        sum(data[:,8]), sum(data[:,9]), step_count)))
    print('Max depth = {}'.format(stat_list(data[:,10])))
    print('Max steer = {}'.format(stat_list(data[:,11])))
    print('Failure = {}'.format(stat_list(data[:, 13])))
