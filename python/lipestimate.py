import numpy as np
import pandas as pd
import os

from eclipseE_fast import eclipseE_fast as eclipse_fast
from eclipseE import eclipseE as eclipse

def load_weights(dirname, layersize, neurons, rdseed):
    weights = np.load(dirname + os.sep + 'lyr' + str(layersize) + 'n' + str(neurons) + 'test' + str(rdseed) + '.npz')
    return weights

dirname = r'datasets'
lyrs = [2]
neurons = [80, 100]

# estimation
data_ini = np.zeros((len(neurons), len(lyrs)))
lipest = np.zeros((len(neurons), len(lyrs)))
timeused = np.zeros((len(neurons), len(lyrs)))
trivialres = np.zeros((len(neurons), len(lyrs)))

for i_lyr, lyr in enumerate(lyrs):
    for i_n, n in enumerate(neurons):
        for i_rd, rd in enumerate([1]):
            weights = load_weights(dirname, lyr, n, rd)
            # lip, trivial, time = eclipse_fast(weights)
            lip, trivial, time = eclipse(weights)

            lipest[i_n, i_lyr] = lip
            timeused[i_n, i_lyr] = time
            trivialres[i_n, i_lyr] = trivial

ratio = lipest / trivialres
print(ratio)
print(lipest)
print(trivialres)
print(timeused)