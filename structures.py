import numpy as np

GLIDERS = [
    # up left
    np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]]),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]]),
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]]),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]]),
    # up right
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]]),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]]),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]),
    # down left
    np.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]]),
    np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0]]),
    np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]]),
    # down right
    np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),
    np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]]),
    np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
    np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]]),
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

STRUCTURE_NAMES = (
    ["glider"] * len(GLIDERS)
    + ["rpentimino"] * len(R_PENTIMINOS)
    + ["exploder"] * len(EXPLODERS)
)
