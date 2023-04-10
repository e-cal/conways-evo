import random

import numpy as np

###############################
# Mutation Algorithms
###############################


def bitflip(chrom: np.ndarray, k=2) -> np.ndarray:
    out_shape = chrom.shape
    chrom = chrom.flatten()

    # choose k indices without replacement
    selected = random.sample([x for x in range(len(chrom))], k=k)

    # flip the selected bits
    for i in selected:
        chrom[i] = 1 - chrom[i]

    return chrom.reshape(out_shape)


def scramble(chrom: np.ndarray) -> np.ndarray:
    return np.random.permutation(chrom.flatten()).reshape(chrom.shape)
