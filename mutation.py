import random
from pprint import pprint

import numpy as np


class Mutation:
    @staticmethod
    def bitflip(chromo: np.ndarray, k=2) -> np.array:
        og_shape = chromo.shape
        chromo = chromo.flatten()

        # flip k indices at random
        selected = random.sample([x for x in range(len(chromo))], k=k)

        for i in selected:
            if chromo[i] == 0:
                chromo[i] = 1
            else:
                chromo[i] = 0

        return np.reshape(chromo, og_shape)

    @staticmethod
    def scramble(chromo: np.array) -> np.array:
        return np.random.permutation(chromo.flatten()).reshape(chromo.shape)


if __name__ == "__main__":
    c = np.random.randint(low=0, high=2, size=(5, 5))
    print("old", c)
    new = Mutation.bitflip(c, k=10)
    print("new", new)

    print("\n-------------------------\n")

    print("old")
    pprint(c)
    new = Mutation.scramble(c)
    print("new")
    pprint(new)
