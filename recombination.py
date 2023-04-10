import random

import numpy as np

###############################
# Recombination Algorithms
###############################


def k_point_crossover(parent1: np.ndarray, parent2: np.ndarray, k=1) -> np.ndarray:
    # pre flight checks
    assert parent1.shape == parent2.shape

    if k == 0:
        return parent1, parent2

    og_shape = parent1.shape

    # flatten the game into a bit string
    c1 = parent1.flatten()
    c2 = parent2.flatten()

    # pick k random indices and sort
    selected = random.sample([x for x in range(len(c1))], k=k)
    selected.sort()

    # get the first cut since we always flip it
    offspring1 = np.array(c1[: selected[0]])
    offspring2 = np.array(c2[: selected[0]])

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
            offspring1 = np.array(np.concatenate((offspring1, cut2)))
            offspring2 = np.array(np.concatenate((offspring2, cut1)))
        else:
            offspring1 = np.array(np.concatenate((offspring1, cut1)))
            offspring2 = np.array(np.concatenate((offspring2, cut2)))

        # set the flag to skip/flip the next section
        flip = not flip

    offspring1 = np.reshape(offspring1, og_shape)
    offspring2 = np.reshape(offspring2, og_shape)

    return offspring1, offspring2
