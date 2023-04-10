from collections import defaultdict

import numpy as np
from scipy.signal import convolve2d

from structures import STRUCTURE_NAMES, STRUCTURES


def soft_hash(mat):
    return np.sum(mat)


def find_structures(structures, search_space) -> list[tuple[int, int]]:
    # Use convolution to find the submatrices more efficiently
    all_matches = []

    # Compute the hash of the search space once
    search_space_hash = convolve2d(
        search_space, np.ones(np.array(structures[0]).shape), mode="valid"
    )

    for target in structures:
        target_hash = soft_hash(target)
        target_rows, target_cols = np.array(target).shape

        # Find the indices where the search_space_hash matches the target_hash
        match_indices = np.where(search_space_hash == target_hash)

        matches = []
        for i, j in zip(match_indices[0], match_indices[1]):
            if np.array_equal(
                search_space[i : i + target_rows, j : j + target_cols], target
            ):
                matches.append((i, j))

        all_matches.append(matches)

    return all_matches


def count_structures(search_space, debug=False):
    found_structures = defaultdict(int)
    locations = find_structures(STRUCTURES, search_space)
    for i, (struct_locs, name) in enumerate(zip(locations, STRUCTURE_NAMES)):
        if struct_locs:
            n = len(struct_locs)
            found_structures[name] += n

            if debug:
                print(f"Found {n} {name}(s) [struct id {i}] at {struct_locs}")

    return dict(found_structures)


def num_structures(cells, debug=False):
    structures = count_structures(cells, debug)
    return sum(structures.values())


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    from structures import EXPLODERS, GLIDERS, R_PENTIMINOS

    NROWS = 60
    NCOLS = 60
    cells = np.zeros((NROWS, NCOLS))

    glider = GLIDERS[0]
    pos = (NROWS // 2, NCOLS // 2)
    cells[pos[0] : pos[0] + glider.shape[0], pos[1] : pos[1] + glider.shape[1]] = glider

    glider = GLIDERS[2]
    pos = ((NROWS // 2) + 10, (NCOLS // 2) + 10)
    cells[pos[0] : pos[0] + glider.shape[0], pos[1] : pos[1] + glider.shape[1]] = glider

    rpent = R_PENTIMINOS[2]
    pos = ((NROWS // 2), (NCOLS // 2) + 10)
    cells[pos[0] : pos[0] + rpent.shape[0], pos[1] : pos[1] + rpent.shape[1]] = rpent

    rpent = R_PENTIMINOS[2]
    pos = ((NROWS // 2) - 10, (NCOLS // 2) - 10)
    cells[pos[0] : pos[0] + rpent.shape[0], pos[1] : pos[1] + rpent.shape[1]] = rpent

    exploder = EXPLODERS[0]
    pos = ((NROWS // 2) + 10, (NCOLS // 2) - 10)
    cells[
        pos[0] : pos[0] + exploder.shape[0], pos[1] : pos[1] + exploder.shape[1]
    ] = exploder

    # print the grid using matplotlib
    # plt.imshow(grid, cmap="gray")
    # plt.show()
    # total = 0
    # for _ in range(1000):
    #     start = time.time()
    #     # total_structures(grid, False)
    #     count_better(grid)
    #     total += time.time() - start
    # print(total / 1000)

    # print("old:", count_structures(grid))
    # print("new:", count_better(grid))

    print(num_structures(cells, True))
