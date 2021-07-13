import numpy as np
import sigfig

class Statistics:
    def __init__(self):
        self.data = []

    def append(self, x):
        self.data.append(x)

    def extend(self, xs):
        self.data.extend(xs)

    def mean(self):
        if len(self.data) == 0:
            return 0.0
        return np.mean(self.data)

    def stderr(self):
        if len(self.data) < 2:
            return 0.0
        return np.std(self.data, ddof=1) / np.sqrt(len(self.data))

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '{:.03f} ({})'.format(self.mean(),
                sigfig.round(float(self.stderr()), sigfigs=2))
