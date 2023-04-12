import argparse
import concurrent.futures as cf
import os

import numpy as np
from tqdm import tqdm, trange

###############################################################################
#                                   Setup
###############################################################################
# Import components of the genetic algorithm
from evaluate import num_structures as evaluate
from mutation import bitflip as mutate
from recombination import k_point_crossover as recombine
from selection import mu_plus_lambda as select_survivors
from selection import nbest as select_parents

# Game
NCOLS = 60
NROWS = 60

# Genetic Algorithm
POP_SIZE = 64
N_PARENTS = 8

MAX_GENS = 400
EVAL_WINDOW = (40, 54)
# end when done evaluating (+1 for non-inclusive ranges)
N_STEPS = EVAL_WINDOW[1] + 1

# Logging
DEFUALT_PATH = "history"

###############################################################################
#                                   Main Functions
###############################################################################


def init():
    cells = np.random.randint(low=0, high=2, size=(POP_SIZE, NROWS, NCOLS))
    return list(cells)


def update(cur):
    nxt = np.zeros((cur.shape[0], cur.shape[1]))  # all cells dead by default

    for row, col in np.ndindex(cur.shape):
        alive_neighbors = (
            np.sum(cur[row - 1 : row + 2, col - 1 : col + 2]) - cur[row, col]
        )

        # a cell will be alive in the next generation if it satisfies one of two conditions:
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
        if EVAL_WINDOW[0] <= step <= EVAL_WINDOW[1]:
            fitness += evaluate(cells)
        cells = update(cells)

    return fitness


def async_run(population, progress_bar=None):
    fitnesses = [0] * len(population)

    with cf.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run, individual): i
            for i, individual in enumerate(population)
        }

        for future in cf.as_completed(futures):
            i = futures[future]
            fitnesses[i] = future.result()

            if progress_bar:
                progress_bar.update(1)

    return fitnesses


def generate_offspring(parent_idxs, population):
    offspring = []
    for _ in range(N_PARENTS // 2):
        # randomly pair parents, popping them from the pool
        parent1 = population[parent_idxs.pop(np.random.randint(len(parent_idxs)))]
        parent2 = population[parent_idxs.pop(np.random.randint(len(parent_idxs)))]

        offspring.extend(recombine(parent1, parent2))
    return offspring


def ga(fp):
    print(f"Starting genetic algorithm with {POP_SIZE} random individuals\n")
    population = init()

    bar_format = "{l_bar}{bar}| eta: {remaining}"

    with tqdm(
        total=len(population),
        desc="Evaluating initial population",
        bar_format=bar_format,
        position=1,
        leave=False,
    ) as progbar:
        fitnesses = async_run(population, progbar)

    # Create a progress bar for the outer loop (generations)
    with trange(
        1,
        MAX_GENS + 1,
        desc=f"Generation 0/{MAX_GENS} (best: {max(fitnesses)}, avg: {np.mean(fitnesses):.2f})",
        bar_format=bar_format,
        position=0,
        leave=False,
    ) as gen_progress:
        for gen in gen_progress:
            # select parents
            parent_idxs = select_parents(population, fitnesses, N_PARENTS)

            # generate offspring
            offspring = generate_offspring(parent_idxs, population)

            # mutate offspring
            offspring = list(map(mutate, offspring))

            # evaluate offspring
            offspring_fitnesses = async_run(offspring)

            # select survivors
            population, fitnesses = select_survivors(
                population, fitnesses, offspring, offspring_fitnesses
            )

            # Update the logging output underneath the progress bar
            gen_progress.set_description(
                f"Generation {gen}/{MAX_GENS} (best={max(fitnesses)}, avg={np.mean(fitnesses):.2f})"
            )

            log_and_save(population, fitnesses, gen, fp)

    print(f"\nCompleted {MAX_GENS} generations.\nBest fitness: {max(fitnesses)}")


def log_and_save(population, fitnesses, gen, fp):
    if not os.path.exists(f"{fp}/log.csv"):
        with open(f"{fp}/log.csv", "w") as f:
            f.write("generation,max,avg,min")

    with open(f"{fp}/log.csv", "a") as f:
        f.write(f"\n{gen},{max(fitnesses)},{np.mean(fitnesses)},{min(fitnesses)}")

    max_fitness = max(fitnesses)
    n = 1
    for individual, fitness in zip(population, fitnesses):
        if fitness == max_fitness:
            np.save(f"{fp}/gen{gen}_{n}.npy", individual)
            n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        help="Path to save logs and individuals (disables input prompt)",
    )
    parser.add_argument(
        "-d",
        "--delete",
        help="Delete the save directory if it already exists (instead of erroring out)",
        action="store_true",
    )
    args = parser.parse_args()

    path = args.filepath
    if path is None:
        path = input(
            f"Enter path to save logs and individuals (default={DEFUALT_PATH}): "
        )
        if path == "":
            path = DEFUALT_PATH

        if path[-1] == "/":
            path = path[:-1]

    if args.delete:
        if os.path.exists(path):
            print(f"Deleting {path}")
            os.system(f"rm -r {path}")
    try:
        os.makedirs(path, exist_ok=False)
    except FileExistsError:
        print(
            f'Directory "{path}" already exists. Please move/delete it or choose a different path.'
        )
        exit()

    ga(path)


if __name__ == "__main__":
    main()
