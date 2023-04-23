import numpy as np

###############################
# Parent Selection Algorithms
###############################


def nbest(population, fitnesses, n):
    """Best n parent selection"""
    idxs = np.array(fitnesses).argsort()[::-1][:n]
    return list(idxs)


def tournament(fitness, mating_pool_size, tournament_size):
    """Tournament selection without replacement"""

    selected_to_mate = []

    # initialize population to select (and remove) from
    pool = list(range(len(fitness)))
    # repeat until mating pool has enough individuals
    while len(selected_to_mate) < mating_pool_size:
        best = None  # init best in the tournament
        for _ in range(tournament_size):  # compare n=tournament_size individuals
            contender = np.random.choice(pool)  # choose a random contender
            # check if the contender is better than the previous best
            if best is None or fitness[contender] > fitness[best]:
                best = contender
        idx = pool.index(best)  # type: ignore
        # add the tournament winner to the selected parents and remove from the pool
        selected_to_mate.append(pool.pop(idx))

    return selected_to_mate


###############################
# Survivor Selection Algorithms
###############################


def mu_plus_lambda(current_pop, current_fitness, offspring, offspring_fitness):
    """mu+lambda selection"""
    population = []
    fitness = []

    # select the mu fittest individuals from both pools
    mu = len(current_pop)
    pool = np.array(current_pop + offspring)
    fp = np.array(current_fitness + offspring_fitness)

    # sort the fitness and pick the top mu choices
    idx = fp.argsort()[::-1][:mu]

    # add the top mu choices to the new population
    for i in range(mu):
        ix = idx[i]
        population.append(pool[ix])
        fitness.append(fp[ix])

    return population, fitness


def replacement(current_pop, current_fitness, offspring, offspring_fitness):
    """replacement selection"""

    population = []
    fitness = []

    mu = len(current_fitness)

    # sort the fitness and replace the worst mu individuals
    idx = np.array(current_fitness).argsort()[::-1][:mu]

    for i in range(mu - len(offspring)):
        ix = idx[i]
        population.append(np.array(current_pop)[ix].tolist())
        fitness.append(np.array(current_fitness)[ix].tolist())

    population += offspring
    fitness += offspring_fitness

    return population, fitness


def crowding_replacement(current_pop, current_fitness, offspring, offspring_fitness):
    """Offspring replaces the most similar from a random subset of the current population"""

    def _distance(child, individual):
        """Euclidean distance between two individuals"""
        return np.linalg.norm(np.array(child) - np.array(individual))

    mu = len(current_pop)
    labmda = len(offspring)
    # choose a random subset of the current population
    idx = np.random.choice(mu, size=labmda * 2, replace=False)

    # to_replace is the individuals from the current population that will be replaced (idxs)
    to_replace = np.array(current_pop)[idx]

    # for each offspring
    # generate a list of distances to each individual in the current population
    distances = [
        [_distance(child, individual) for individual in to_replace]
        for child in offspring
    ]

    # map individual index to offspring that will replace it
    replacements = {}
    for i, child in enumerate(offspring):
        # find the index of the most similar individual
        ix = np.argmin(distances[i])

        taken = ix in replacements
        while taken:
            # if the index is already taken, find the next best
            distances[i][ix] = np.inf  # type: ignore
            ix = np.argmin(distances[i])
            taken = ix in replacements

        replacements[ix] = i, child

    for i, (ci, child) in replacements.items():
        current_pop[i] = child
        current_fitness[i] = offspring_fitness[ci]

    return current_pop, current_fitness
