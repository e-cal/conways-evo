import random
from pprint import pprint

import numpy as np


###############################
# Recombination Algorithms
###############################
def k_point_crossover(c1: np.ndarray, c2: np.ndarray, k=1) -> np.ndarray:
    # pre flight checks
    assert c1.shape == c2.shape

    if k == 0:
        return c1, c2

    og_shape = c1.shape

    # flatten the game into a bit string
    c1 = c1.flatten()
    c2 = c2.flatten()

    # pick k random indices and sort
    selected = random.sample([x for x in range(len(c1))], k=k)
    selected.sort()

    # get the first cut since we always flip it
    o1 = np.array(c1[: selected[0]])
    o2 = np.array(c2[: selected[0]])

    flip = True
    for i in range(len(selected)):
        # grab the cuts from the parents
        if i == len(selected) - 1:
            cut1 = np.array(c1[selected[i] :])
            cut2 = np.array(c2[selected[i] :])
        else:
            cut1 = np.array(c1[selected[i] : selected[i + 1]])
            cut2 = np.array(c2[selected[i] : selected[i + 1]])

        # apply the flip every other cut
        if flip:
            o1 = np.array(np.concatenate((o1, cut2)))
            o2 = np.array(np.concatenate((o2, cut1)))
        else:
            o1 = np.array(np.concatenate((o1, cut1)))
            o2 = np.array(np.concatenate((o2, cut2)))

        # set the flag to skip/flip the next section
        flip = not flip

    o1 = np.reshape(o1, og_shape)
    o2 = np.reshape(o2, og_shape)

    return o1, o2
