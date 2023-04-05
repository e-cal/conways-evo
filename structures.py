import numpy as np

GLIDERS = [
    # up left
    np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]]),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]]),
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]]),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]]),  # 3
    # up right
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]]),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]]),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]),  # 7
    # down left
    np.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
    np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]]),
    np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]]),  # 11
    # down right
    np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),
    np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]]),
    np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
    np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]),  # 15
]

R_PENTIMINOS = [
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),  # upright
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]]),  # upright flipped
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]]),  # upside down
    np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]]),  # upside down flipped
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]]),  # right
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]),  # right flipped
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]]),  # left
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]]),  # left flipped
]

EXPLODERS = [
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]),  # up
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),  # right
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]]),  # down
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]]),  # left
]

# e.g. GLIDERS[0] = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])
STRUCTURES = GLIDERS + R_PENTIMINOS + EXPLODERS

# currently all arrays are 3x3
# pad all arrays with 0s to make each array 5x5
STRUCTURES = [np.pad(s, 1, "constant") for s in STRUCTURES]

STRUCTURE_NAMES = (
    ["glider"] * len(GLIDERS)
    + ["rpentimino"] * len(R_PENTIMINOS)
    + ["exploder"] * len(EXPLODERS)
)
