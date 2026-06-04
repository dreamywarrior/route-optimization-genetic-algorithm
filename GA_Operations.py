"""
Genetic Algorithm for Route Optimization

This module implements a Genetic Algorithm (GA) to find optimized routes for visiting cities.
It uses rank-based selection, ordered crossover, and swap mutation to evolve a population
of candidate solutions towards finding routes with minimal travel distance and time.

Key concepts:
- Population: A set of candidate routes/solutions
- Fitness: Evaluated based on total distance and time to complete the route
- Selection: Rank-based selection (entire population ranked, probability proportional to rank)
- Crossover: Ordered crossover to create offspring from parent routes
- Mutation: Swap mutation to introduce variation in the population
"""

import random
import logging
import os
from datetime import datetime

"""
Genetic Algorithm Parameters

These constants control the behavior and performance of the genetic algorithm:
- POPULATION_SIZE: Number of candidate solutions in each generation. Larger values = more diversity but slower computation
- GENERATIONS: Maximum number of iterations the GA runs for
- SPEED: Average travel speed (km/h) - used to calculate total travel time from distance
- MUTATION_RATE: Probability of mutation applied to offspring. Higher rate = more variation, may prevent convergence
- SEED_VALUE: Random seed for reproducibility of results
- ELITISM_COUNT: Number of best individuals carried over unchanged to the next generation
"""

POPULATION_SIZE = 120
GENERATIONS = 500
SPEED = 7.5
MUTATION_RATE = 0.3
SEED_VALUE = 42
ELITISM_COUNT = 2

def is_valid_route(route):
    """
    Validates that a route is a valid solution to the TSP.

    A valid route must:
    1. Have exactly 21 elements (19 cities + start and end depot)
    2. Start at city 0
    3. End at city 0
    4. Visit each city 1-19 exactly once

    Args:
        route (list): A candidate route to validate

    Returns:
        bool: True if route is valid, False otherwise
    """
    return len(route) == 21 and route[0] == 0 and route[-1] == 0 and set(route[1:-1]) == set(range(1, 20))

def generate_city_coordinates():
    """
    Generates random geographical coordinates for 20 cities.

    Why: Simulates a real TSP scenario where cities have 2D positions.
    The coordinates are used to calculate Euclidean distances between cities.
    City 0 is the starting point, cities 1-19 are regular cities to visit.

    Returns:
        dict: Mapping of city ID to [x, y] coordinates (x, y both in range 1-100)
    """
    coordinates = {}
    for i in range(20):
        coordinates[i] = [random.randint(1, 100), random.randint(1, 100)]
    return coordinates

def generate_route():
    """
    Creates a random candidate route (solution)

    Why: Initial population must have diverse solutions. Random generation ensures
    the starting population has different genetic material for evolution to work.
    Each route visits all cities 1-19 in random order, starting and ending at city 0.

    Returns:
        list: A valid route starting and ending at city 0 with cities 1-19 in random order
    """
    route = list(range(1, 20))
    random.shuffle(route)
    return [0] + route + [0]

def populate_population():
    """
    Creates the initial population of random routes.

    Why: Genetic algorithms start with a diverse population. Larger populations
    have more genetic diversity but require more computation per generation.

    Returns:
        list: Population of POPULATION_SIZE valid routes
    """
    population = [generate_route() for _ in range(POPULATION_SIZE)]
    return population

def fitness_function(route):
    """
    PLACEHOLDER FITNESS FUNCTION - Evaluates the quality of a route solution.

    Why: Fitness determines which routes are "good" (short) vs "bad" (long).
    Better fitness routes are more likely to be selected for breeding, guiding
    evolution towards optimal solutions.

    The fitness is calculated as:
    1. Euclidean distance between consecutive cities in the route
    2. Total time = total distance / SPEED

    Args:
        route (list): A route to evaluate

    Returns:
        list: [total_distance (km), total_time (hours)]

    Note: Lower fitness values are better (shorter routes are fitter)
    """
    total_distance = 0
    for i in range(len(route) - 1):
        city_a = route[i]
        city_b = route[i + 1]
        distance = ((city_coordinates[city_a][0] - city_coordinates[city_b][0])**2 + 
                    (city_coordinates[city_a][1] - city_coordinates[city_b][1])**2) ** 0.5
        total_distance += distance
    total_time = total_distance / SPEED
    return [round(total_distance, 2), round(total_time, 2)]


def build_rank_probabilities(pop_size):
    """
    Pre-computes a cumulative probability array for rank-based selection.

    Why: In rank selection, every individual is assigned a rank based on
    its position in the sorted population. The worst gets rank 1, the best
    gets rank N. Selection probability is proportional to rank, so better
    individuals are more likely to be picked — but even the worst still has
    a small chance, which helps maintain diversity.

    We compute this once per generation and reuse it for every parent
    selection in that generation, instead of recomputing each time.

    Args:
        pop_size (int): Number of individuals in the population

    Returns:
        list: Cumulative probability array of length pop_size.
              Entry i corresponds to the individual at index i in a
              population sorted ascending by distance (index 0 = worst,
              index N-1 = best).
    """
    # rank 1 for worst (index 0), rank N for best (index N-1)
    total_rank = pop_size * (pop_size + 1) / 2

    cumulative = []
    running = 0.0
    for rank in range(1, pop_size + 1):
        running += rank / total_rank
        cumulative.append(running)

    logging.info(f"Rank probabilities built for {pop_size} individuals. "
                 f"Worst prob: {1/total_rank:.6f}, Best prob: {pop_size/total_rank:.6f}")
    return cumulative


def rank_selection(sorted_population, cumulative):
    """
    Selects two parent routes from the population using rank-based selection.

    Why: Rank selection avoids problems that raw fitness-proportionate selection
    has — like one super-fit individual dominating all selections, or tiny
    fitness differences making selection nearly random. By using ranks instead
    of raw fitness values, the selection pressure stays consistent regardless
    of how close or far apart the actual distances are.

    How it works:
    1. Population must already be sorted ascending by distance (worst first,
       best last). This sorting is done once per generation in the GA loop.
    2. A cumulative probability array (pre-built by build_rank_probabilities)
       maps each position to its selection probability based on rank.
    3. To pick a parent, generate a random number in [0,1) and walk the
       cumulative array to find which individual it falls on.
    4. Repeat to pick a second parent, ensuring it's not the same individual.

    Args:
        sorted_population (list): Population sorted ascending by distance
        cumulative (list): Pre-computed cumulative probability array

    Returns:
        list: Two selected parent routes [parent1, parent2]
    """
    parents = []
    selected_indices = set()

    while len(parents) < 2:
        rand = random.random()
        for idx, threshold in enumerate(cumulative):
            if rand <= threshold:
                if idx not in selected_indices:
                    parents.append(sorted_population[idx])
                    selected_indices.add(idx)
                    logging.info(f"Rank selection picked index {idx} "
                                 f"(rank {idx+1}/{len(sorted_population)}), "
                                 f"rand={rand:.4f}, threshold={threshold:.6f}")
                break

    return parents


def ordered_crossover(parents):
    """
    Creates offspring by combining genetic material from two parent routes.

    Why: Crossover combines the good traits of both parents, potentially creating
    better offspring. Ordered crossover (OX) is specifically designed for TSP
    to maintain tour validity (no repeated cities).

    How Ordered Crossover works:
    1. Select two random cut points in the route
    2. Copy the segment between cut points from parent1 to child
    3. Fill remaining positions with parent2's cities in order (wrapping around)
    4. This preserves good orderings from both parents while maintaining validity

    Args:
        parents (list): Two parent routes [parent1, parent2]

    Returns:
        list: One offspring route that combines parent genes
    """
    parent1, parent2 = parents
    logging.info(f"Parent 1: {parent1}")
    logging.info(f"Parent 2: {parent2}")
    cut1, cut2 = sorted(random.sample(range(1,19), 2))
    logging.info(f"Crossover points: {cut1}, {cut2}")

    child = [None] * 19
    child[cut1:cut2+1] = parent1[cut1:cut2+1]
    logging.info(f"Child after copying segment from Parent 1: {child}")

    parent2 = parent2[1:-1]
    start = cut2 + 1
    ordered_parent2 = (parent2[start:] + parent2[:start])
    remaining_genes = [gene for gene in ordered_parent2 if gene not in child]
    fill_positions = (list(range(cut2 + 1, 19)) + list(range(0, cut1)))

    for pos, gene in zip(fill_positions, remaining_genes):
        child[pos] = gene
    child = [0] + child + [0]

    assert is_valid_route(child), f"Invalid route generated: {child}"
    return child

def swap_mutation(child):
    """
    Introduces random variation into offspring by swapping two cities.

    Why: Mutation prevents the population from converging prematurely to a local
    optimum by introducing new genetic variation. It allows exploration of the
    solution space beyond what crossover alone can reach.

    How it works:
    1. Generate random number
    2. If random number < MUTATION_RATE: randomly swap two cities in the route
    3. Otherwise: keep child unchanged

    Lower MUTATION_RATE = less variation (faster convergence, risk of local optima)
    Higher MUTATION_RATE = more variation (slower convergence, explores more)

    Args:
        child (list): Offspring route to potentially mutate

    Returns:
        list: Mutated or unchanged child route
    """
    random_value = random.random()
    logging.info(f"Random value for mutation: {random_value}")
    
    if random_value > MUTATION_RATE:
        logging.info("No mutation applied.")
        return child
    
    logging.info("Mutation applied.")
    idx1, idx2 = random.sample(range(1, 20), 2)
    child[idx1], child[idx2] = child[idx2], child[idx1]

    assert is_valid_route(child), f"Invalid route after mutation: {child}"
    return child

if __name__ == "__main__":

    # Setup logging with timestamp
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"GA_Operations_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler()
        ]
    )
    
    random.seed(SEED_VALUE)

    city_coordinates = generate_city_coordinates()
    logging.info(f"City coordinates: {city_coordinates}")
    
    population = populate_population()

    # -------------------------------------------------------------------
    # Main GA Loop with Elitism and Convergence Tracking
    #
    # Each generation:
    #   1. Sort population by distance ascending (worst first, best last)
    #   2. Build rank probability wheel once for this generation
    #   3. Track the best distance and time for convergence analysis
    #   4. Copy the top ELITISM_COUNT individuals directly into next gen
    #   5. Fill remaining slots using rank selection -> crossover -> mutation
    #   6. Replace old population with the new one
    # -------------------------------------------------------------------

    best_distances = []   # tracks best distance found at each generation
    best_times = []       # tracks best time found at each generation

    for gen in range(GENERATIONS):

        # sort population by total distance (fitness_function returns [dist, time])
        # ascending: index 0 = worst (longest), index -1 = best (shortest)
        population.sort(key=lambda route: fitness_function(route)[0], reverse=True)

        # build the rank probability wheel once for this generation
        cumulative = build_rank_probabilities(len(population))

        # last individual in sorted list is the current best (shortest distance)
        current_best = population[-1]
        current_best_fitness = fitness_function(current_best)
        best_distances.append(current_best_fitness[0])
        best_times.append(current_best_fitness[1])

        logging.info(f"Generation {gen+1}: Best Distance = {current_best_fitness[0]}, Best Time = {current_best_fitness[1]}")

        # --- Elitism: carry over the top ELITISM_COUNT routes unchanged ---
        # best routes are at the end since we sorted ascending (worst first)
        new_population = []
        for i in range(ELITISM_COUNT):
            elite = population[-(i+1)]
            new_population.append(elite)
            logging.info(f"Elite {i+1} preserved: fitness = {fitness_function(elite)}")

        # --- Fill the rest through rank selection + crossover + mutation ---
        while len(new_population) < POPULATION_SIZE:
            parents = rank_selection(population, cumulative)
            logging.info(f"Rank selection winners: {fitness_function(parents[0])}, {fitness_function(parents[1])}")

            child = ordered_crossover(parents)
            logging.info(f"Child after crossover: {child}")

            child = swap_mutation(child)
            logging.info(f"Child after mutation: {child}, fitness: {fitness_function(child)}")

            new_population.append(child)

        population = new_population

        # print progress every 50 generations
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"Gen {gen+1:>4d} | Best Distance: {best_distances[-1]:.2f}  |  Best Time: {best_times[-1]:.2f}")

    # --- Final results after all generations ---
    population.sort(key=lambda route: fitness_function(route)[0])
    best_route = population[0]
    best_fitness = fitness_function(best_route)

    print("\n" + "=" * 55)
    print("GA Run Complete")
    print("=" * 55)
    print(f"Best Route: {' -> '.join(str(c) for c in best_route)}")
    print(f"Total Distance: {best_fitness[0]:.2f}")
    print(f"Total Time:     {best_fitness[1]:.2f} hours")
    print(f"\nConvergence:")
    print(f"  Distance: {best_distances[0]:.2f} -> {best_distances[-1]:.2f}")
    print(f"  Time:     {best_times[0]:.2f} -> {best_times[-1]:.2f}")
    print("=" * 55)

    logging.info(f"Final Best Route: {best_route}")
    logging.info(f"Final Best Fitness: {best_fitness}")
    logging.info(f"Distance convergence: {best_distances[0]} -> {best_distances[-1]}")
    logging.info(f"Time convergence: {best_times[0]} -> {best_times[-1]}")
