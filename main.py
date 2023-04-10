import argparse
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
MAX_GENS = 3  # 400
N_STEPS = EVAL_WINDOW[1]  # end when done evaluating

DEFUALT_PATH = "history"


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


def main(fp):
    print(f"Starting genetic algorithm with {POP_SIZE} random individuals\n")
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

        log(fitnesses, gen, fp)
        save_best(population, fitnesses, gen, fp)


def log(fitnesses, gen, fp):
    print(
        f"Generation {gen}\n  Best fitness: {max(fitnesses)}\n  Avg fitness: {np.mean(fitnesses):.2f}\n  Worst fitness: {min(fitnesses)}"
    )
    if not os.path.exists(f"{fp}/log.txt"):
        with open(f"{fp}/log.txt", "w") as f:
            f.write("generation,max,avg,min")

    with open(f"{fp}/log.txt", "a") as f:
        f.write(f"\n{gen},{max(fitnesses)},{np.mean(fitnesses)},{min(fitnesses)}")


def save_best(population, fitnesses, gen, fp):
    max_fitness = max(fitnesses)
    n = 1
    for individual, fitness in zip(population, fitnesses):
        if fitness == max_fitness:
            np.save(f"{fp}/gen{gen}_{n}.npy", individual)
            n += 1

    print(f"Saved {n - 1} individual(s) to {fp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="path to save logs and individuals")
    parser.add_argument(
        "-d", "--delete", help="delete existing directory", action="store_true"
    )
    args = parser.parse_args()

    fp = args.file
    if fp is None:
        fp = input(
            f"Enter path to save logs and individuals (default={DEFUALT_PATH}): "
        )
        if fp == "":
            fp = DEFUALT_PATH

        if fp[-1] == "/":
            fp = fp[:-1]

    if args.delete:
        if os.path.exists(fp):
            print(f"Deleting {fp}")
            os.system(f"rm -r {fp}")
    try:
        os.makedirs(fp, exist_ok=False)
    except FileExistsError:
        print(
            f'Directory "{fp}" already exists. Please move/delete it or choose a different path.'
        )
        exit()

    main(fp)
