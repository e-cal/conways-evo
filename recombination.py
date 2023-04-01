import numpy as np
import random
from pprint import pprint

class Recombination:

    @staticmethod
    def k_point_crossover(c1: np.array, c2: np.array, k=2) -> np.array:
        
        # pre flight checks
        assert(c1.shape == c2.shape)

        if k == 0:
            return c1, c2
        
        og_shape = c1.shape

        # flatten the game into a bit string
        c1 = c1.flatten()
        c2 = c2.flatten()

        # pick k random indices and sort
        selected = random.choices([x for x in range(len(c1))], k=k)
        selected.sort()

        # get the first cut since we always flip it
        o1 = np.array(c1[:selected[0]])
        o2 = np.array(c2[:selected[0]])

        flip = True
        for i in range(len(selected)):

            # grab the cuts from the parents
            if i == len(selected) - 1:
                cut1 = np.array(c1[selected[i]:])
                cut2 = np.array(c2[selected[i]:])
            else:
                cut1 = np.array(c1[selected[i]:selected[i+1]])
                cut2 = np.array(c2[selected[i]:selected[i+1]])
            
            # apply the flip every other cut
            if flip:
                o1 = np.array(np.concatenate((o1,cut2)))
                o2 = np.array(np.concatenate((o2,cut1)))
            else:
                o1 = np.array(np.concatenate((o1,cut1)))
                o2 = np.array(np.concatenate((o2,cut2)))

            # set the flag to skip/flip the next section
            flip = not flip
            
        o1 = np.reshape(o1, og_shape)
        o2 = np.reshape(o2, og_shape)

        return o1, o2


if __name__ == '__main__':

    c1 = np.random.randint(low=0, high=2, size=(60, 60))
    c2 = np.random.randint(low=0, high=2, size=(60, 60))

    print('c1')
    pprint(c1)
    print('c2')
    pprint(c2)

    o1, o2 = Recombination.k_point_crossover(c1, c2, k=8)

    print('-----------------------------')

    print('o1')
    pprint(o1)
    print('o2')
    pprint(o2)





