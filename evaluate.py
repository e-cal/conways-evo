from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from structures import *


def _hash(mat):
    return np.sum(mat)


def find_submatrices(target, search_space) -> list[tuple[int, int]]:
    target_hash = _hash(target)
    target_rows, target_cols = target.shape
    search_rows, search_cols = search_space.shape

    match_matrix = np.zeros(
        (search_rows - target_rows + 1, search_cols - target_cols + 1)
    )

    for i in range(search_rows - target_rows + 1):
        for j in range(search_cols - target_cols + 1):
            sub_matrix = search_space[i : i + target_rows, j : j + target_cols]
            match_matrix[i, j] = _hash(sub_matrix)

    match_indices = np.where(match_matrix == target_hash)

    matches = []
    for i, j in zip(match_indices[0], match_indices[1]):
        if np.array_equal(
            search_space[i : i + target_rows, j : j + target_cols], target
        ):
            matches.append((i, j))

    return matches


def fast_find_submatrices(target, search_space) -> list[tuple[int, int]]:
    target_hash = _hash(target)
    target_rows, target_cols = target.shape

    # Use convolution to find the submatrices more efficiently
    conv_mat = convolve2d(
        search_space, np.flip(np.flip(target, axis=0), axis=1), mode="valid"
    )
    match_indices = np.where(conv_mat == target_hash)

    matches = []
    for i, j in zip(match_indices[0], match_indices[1]):
        if np.array_equal(
            search_space[i : i + target_rows, j : j + target_cols], target
        ):
            matches.append((i, j))

    return matches


def count_structures(search_space, debug=False) -> dict[str, int]:
    found_structures = defaultdict(int)
    for i, (name, structure) in enumerate(zip(STRUCTURE_NAMES, STRUCTURES)):
        positions = fast_find_submatrices(structure, search_space)
        if positions:
            n = len(positions)
            found_structures[name] += n

            if debug:
                print(f"Found {n} {name}(s) [struct id {i}] at {positions}")

    return dict(found_structures)


def evaluate(cells, debug=False):
    structures = count_structures(cells, debug)
    return sum(structures.values())


def init():
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

    return cells


if __name__ == "__main__":
    grid = init()
    # print the grid using matplotlib
    plt.imshow(grid, cmap="gray")
    plt.show()

    print(evaluate(grid, True))
