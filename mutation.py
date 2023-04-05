import random
from pprint import pprint

import numpy as np


def mutate(chrom: np.ndarray, k=2) -> np.ndarray:
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


if __name__ == "__main__":
    c = np.random.randint(low=0, high=2, size=(5, 5))
    print("old", c)
    new = mutate(c, k=2)
    print("new", new)

    print("\n-------------------------\n")

    print("old")
    pprint(c)
    new = scramble(c)
    print("new")
    pprint(new)
