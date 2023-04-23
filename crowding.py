from main import (DEFUALT_PATH, argparse, async_run, init, mutate, np, os,
                  recombine, select_parents, tqdm, trange)
from selection import crowding_replacement as select_survivors

POP_SIZE = 64
MAX_GENS = 400

EVAL_WINDOW = (40, 54)
N_STEPS = EVAL_WINDOW[1] + 1

N_PARENTS = int(0.2 * POP_SIZE)  # ~20% of the population

###############################################################################
#                                   Main Functions
###############################################################################


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

    idxs = np.random.choice(len(population), 3, replace=False)

    max_fitness = max(fitnesses)
    n = 1
    for i, (individual, fitness) in enumerate(zip(population, fitnesses)):
        if fitness == max_fitness:
            np.save(f"{fp}/gen{gen}_best{n}.npy", individual)
            n += 1
        elif i in idxs:
            np.save(f"{fp}/gen{gen}_rand{i}.npy", individual)


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
