import os
import time

import numpy as np

from evaluate import evaluate
from mutation import mutate

# Game
NCOLS = 60
NROWS = 60

# Genetic Algorithm
POP_SIZE = 3  # 64
EVAL_WINDOW = (40, 54)
MAX_GENS = 1  # 400
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
    population = init()
    fitnesses = [0] * POP_SIZE

    for gen in range(MAX_GENS):
        for i, individual in enumerate(population):
            fitnesses[i] = run(individual)

        # select parents
        # parents =

        # generate offspring
        # offspring =

        # mutate offspring

        # evaluate offspring
        # offspring_fitnesses = [0] * len(offspring)
        # for i, individual in enumerate(offspring):
        #     offspring_fitnesses[i] = run(individual)

        # select survivors

        print(
            f"Generation {gen}\n  Best fitness: {max(fitnesses)}\n  Avg fitness: {np.mean(fitnesses)}"
        )

    # save the best individuals to a file
    max_fitness = max(fitnesses)
    i = 1
    for individual, fitness in zip(population, fitnesses):
        if fitness == max_fitness:
            os.makedirs("best", exist_ok=True)
            np.save(f"best/{i}.npy", individual)
            i += 1


if __name__ == "__main__":
    main()