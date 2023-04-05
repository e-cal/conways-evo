import time

import numpy as np

from evaluate import evaluate
from mutation import Mutation

# Game
NCOLS = 60
NROWS = 60

# Genetic Algorithm
POP_SIZE = 64
EVAL_WINDOW = (40, 54)
MAX_GENS = 400
N_STEPS = EVAL_WINDOW[1]  # end when done evaluating


def init():
    cells = np.random.randint(low=0, high=2, size=(POP_SIZE, NROWS, NCOLS))
    return cells


def update(cur):
    nxt = np.zeros((cur.shape[0], cur.shape[1]))  # all cells dead by default

    for row, col in np.ndindex(cur.shape):
        alive_neighbors = (
            np.sum(cur[row - 1 : row + 2, col - 1 : col + 2]) - cur[row, col]
        )

        # cell alive in the next generation under two conditions:
        # survival: cell was alive and has 2 or 3 alive neighbors
        # reproduction: cell was dead and has 3 alive neighbors
        if (
            (cur[row, col] == 1 and 2 <= alive_neighbors <= 3) or 
            (cur[row, col] == 0 and alive_neighbors == 3)  # fmt: skip
        ):
            nxt[row, col] = 1

    return nxt


def run(cells):
    fitness = 0
    for step in range(N_STEPS):
        cells = update(cells)
        if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
            fitness += evaluate(cells)

    return fitness


def main():
    # init
    population = init()
    fitnesses = [0] * POP_SIZE

    for gen in range(MAX_GENS):
        for i, individual in enumerate(population):
            start = time.time()
            fitnesses[i] = run(individual)
            print(f"Gen {gen} individual {i} fitness {fitnesses[i]}")
            print(f"Took: {time.time() - start:.2f}s")

        print(fitnesses)
        break
    # chrom = Selection.tournament(fitnesses, POP_SIZE, chrom)
    # print(chrom)
    # print(f"Generation {gen} done")

    # 1 gen


if __name__ == "__main__":
    main()
