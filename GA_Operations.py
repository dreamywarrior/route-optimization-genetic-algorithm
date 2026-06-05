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

class RouteOptimizationEnvironment:
    """Environment to handle input parsing, coordinates, and fitness logic.

    The class reads `Tmax` and `Cmax` and the city coordinate table from the
    provided input file. It exposes `calculate_fitness(route)` which returns a
    tuple `(fitness_score, total_distance, total_time_hours)` where larger
    fitness_score is better (so sorting ascending places worst first, best
    last).
    """
    def __init__(self, input_filepath="inputPS15.txt", speed=SPEED):
        self.input_filepath = input_filepath
        self.speed = speed
        self.Tmax = None
        self.Cmax = None
        self.city_coordinates = {}
        self._load_input_file(input_filepath)

    def _load_input_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Expect first two non-empty lines to be Tmax and Cmax
        try:
            tline = lines[0]
            cline = lines[1]
            self.Tmax = float(tline.split('=')[1])
            self.Cmax = float(cline.split('=')[1])
        except Exception as e:
            logging.error(f"Failed to parse Tmax/Cmax from {path}: {e}")
            raise

        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            self.city_coordinates[idx] = [x, y]

        logging.info(f"Loaded input: Tmax={self.Tmax}, Cmax={self.Cmax}, "
                     f"{len(self.city_coordinates)} cities from {path}")

    def calculate_distance(self, route):
        total_distance = 0.0
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            xa, ya = self.city_coordinates[a]
            xb, yb = self.city_coordinates[b]
            total_distance += ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
        return round(total_distance, 2)

    def calculate_time(self, distance):
        return round(distance / self.speed, 2)

    def calculate_fitness(self, route):
        """Return (fitness_score, total_distance, total_time_hours).

        Fitness is converted from a cost so that larger is better. Cost is
        distance + time-weight + heavy penalties for constraint violations.
        """
        total_distance = self.calculate_distance(route)
        total_time_hours = total_distance / self.speed

        penalty = 0.0
        if self.Cmax is not None and total_distance > self.Cmax:
            penalty += (total_distance - self.Cmax) * 100.0
        if self.Tmax is not None and total_time_hours > self.Tmax:
            penalty += (total_time_hours - self.Tmax) * 100.0

        cost = total_distance + (total_time_hours * 10.0) + penalty
        fitness_score = 1.0 / (1.0 + cost)
        return (fitness_score, round(total_distance, 2), round(total_time_hours, 2))

    def write_output(self, best_route, best_dist, best_time, fitness,
                     pop_size, generations, output_filepath="outputPS15.txt"):
        lines = []
        lines.append(f"Best Route: {' -> '.join(str(c) for c in best_route)}")
        lines.append(f"Total Distance: {best_dist:.2f}")
        lines.append(f"Total Time: {best_time:.2f} hours")
        dist_status = "SATISFIED" if best_dist <= self.Cmax else "VIOLATED"
        time_status = "SATISFIED" if best_time <= self.Tmax else "VIOLATED"
        lines.append(f"Distance <= Cmax: {dist_status}")
        lines.append(f"Time <= Tmax: {time_status}")
        lines.append(f"Fitness: {fitness:.6f}")
        lines.append("")
        lines.append(f"Population Size: {pop_size}")
        lines.append(f"Generations: {generations}")

        with open(output_filepath, 'w') as f:
            f.write('\n'.join(lines))
        logging.info(f"Wrote output file: {output_filepath}")

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
    # Setup logging
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"GA_Operations_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename)]
    )
    
    random.seed(SEED_VALUE)

    # Initialize the Environment (handles I/O, coordinates, and fitness)
    env = RouteOptimizationEnvironment(input_filepath="inputPS15.txt")
    
    population = populate_population()

    best_distances = []
    best_times = []

    for gen in range(GENERATIONS):

        # Sort ascending by fitness score (worst fitness first, best fitness last)
        population.sort(key=lambda route: env.calculate_fitness(route)[0])

        cumulative = build_rank_probabilities(len(population))

        current_best = population[-1]
        
        # Unpack the new fitness tuple
        current_best_score, current_best_dist, current_best_time = env.calculate_fitness(current_best)
        best_distances.append(current_best_dist)
        best_times.append(current_best_time)

        logging.info(f"Generation {gen+1}: Best Distance = {current_best_dist}, Best Time = {current_best_time}")

        new_population = []
        for i in range(ELITISM_COUNT):
            elite = population[-(i+1)]
            new_population.append(elite)

        while len(new_population) < POPULATION_SIZE:
            parents = rank_selection(population, cumulative)
            child = ordered_crossover(parents)
            child = swap_mutation(child)
            new_population.append(child)

        population = new_population

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"Gen {gen+1:>4d} | Best Distance: {best_distances[-1]:.2f}  |  Best Time: {best_times[-1]:.2f}")

    # Final results
    population.sort(key=lambda route: env.calculate_fitness(route)[0])
    best_route = population[-1]
    best_score, best_dist, best_time = env.calculate_fitness(best_route)

    print("\n" + "=" * 55)
    print("GA Run Complete")
    print("=" * 55)
    print(f"Best Route: {' -> '.join(str(c) for c in best_route)}")
    print(f"Total Distance: {best_dist:.2f}")
    print(f"Total Time:     {best_time:.2f} hours")
    print(f"\nConvergence:")
    print(f"  Distance: {best_distances[0]:.2f} -> {best_distances[-1]:.2f}")
    print(f"  Time:     {best_times[0]:.2f} -> {best_times[-1]:.2f}")
    print("=" * 55)

    logging.info(f"Final Best Route: {best_route}")
    logging.info(f"Final Best Fitness: {(best_score, best_dist, best_time)}")
    logging.info(f"Distance convergence: {best_distances[0]} -> {best_distances[-1]}")
    logging.info(f"Time convergence: {best_times[0]} -> {best_times[-1]}")

    # Generate the strict required output file
    env.write_output(
        best_route=best_route,
        best_dist=best_dist,
        best_time=best_time,
        fitness=best_score,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS
    )
