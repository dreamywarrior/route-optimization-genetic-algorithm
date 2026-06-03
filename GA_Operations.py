"""
Genetic Algorithm for Route Optimization

This module implements a Genetic Algorithm (GA) to find optimized routes for visiting cities.
It uses tournament selection, ordered crossover, and swap mutation to evolve a population
of candidate solutions towards finding routes with minimal travel distance and time.

Key concepts:
- Population: A set of candidate routes/solutions
- Fitness: Evaluated based on total distance and time to complete the route
- Selection: Tournament-based selection with roulette wheel
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
- GENERATIONS: Maximum number of iterations (not used in current implementation but available for full GA loop)
- TOURNAMENT_SIZE: Number of individuals selected for each tournament (fitness competition)
- WINNER_PER_TOURNAMENT: Number of winners selected from each tournament for breeding
- SPEED: Average travel speed (km/h) - used to calculate total travel time from distance
- MUTATION_RATE: Probability of mutation applied to offspring. Higher rate = more variation, may prevent convergence
- SEED_VALUE: Random seed for reproducibility of results
"""

POPULATION_SIZE = 120
GENERATIONS = 500
TOURNAMENT_SIZE = 5
WINNER_PER_TOURNAMENT = 2
SPEED = 7.5
MUTATION_RATE = 0.3
SEED_VALUE = 42

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

def roulette_wheel_selection(tournament_contestants):
    """
    Selects WINNER_PER_TOURNAMENT individuals from tournament using fitness-proportionate selection.

    Why: This method gives better fitness individuals higher probability of being selected,
    mimicking natural selection where fitter organisms are more likely to reproduce.
    Ranking-based approach avoids issues with large fitness value gaps.

    How it works:
    1. Rank contestants (best=highest rank, worst=lowest)
    2. Convert ranks to selection probabilities
    3. Use cumulative probability to make selections via random drawing

    Args:
        tournament_contestants (list): Sorted list of routes from tournament

    Returns:
        list: WINNER_PER_TOURNAMENT selected routes (parents for crossover)
    """
    # Generate cumulative probabilities based on tournament ranks
    ranks = list(range(len(tournament_contestants), 0, -1))
    total_rank = sum(ranks)
        
    probabilities = [r / total_rank for r in ranks]
    cumulative = []
    running = 0
    for p in probabilities:
        running += p
        cumulative.append(running)
    logging.info(f"Cumulative Probabilities: {cumulative}")
    
    # Select a random number and determine the winner based on cumulative probabilities
    winner_list = []
    while len(winner_list) < WINNER_PER_TOURNAMENT:
        rand = random.random()
        logging.info(f"Random number for selection: {rand}")
        for i, threshold in enumerate(cumulative):
             if rand < threshold:
                selected_winner = tournament_contestants[i]
                if selected_winner not in winner_list:
                    winner_list.append(tournament_contestants[i])
                    logging.info(f"Selected winner: {tournament_contestants[i]} with cumulative threshold: {threshold}")
                    break
    return winner_list

def tournament_selection(population, fitness_function):
    """
    Selects the best individuals from a random tournament.

    Why: Tournament selection is computationally efficient and helps maintain diversity
    by not always selecting the global best. It provides a balance between selection
    pressure and population diversity.

    How it works:
    1. Randomly sample TOURNAMENT_SIZE individuals from population
    2. Sort by fitness (lower = better)
    3. Use roulette wheel selection to pick winners from this tournament

    Args:
        population (list): Current population of routes
        fitness_function (function): Function to evaluate route quality

    Returns:
        list: WINNER_PER_TOURNAMENT selected parents for breeding
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness_function, reverse=False)
    logging.info("Tournament contestants and their fitness:")
    for contestant in tournament:
        logging.info(f"Route: {contestant}, Fitness: {fitness_function(contestant)}")

    return roulette_wheel_selection(tournament)

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
    parents = tournament_selection(population, fitness_function)
    logging.info(f"Tournament winners: {parents}")
    
    child = ordered_crossover(parents)
    logging.info(f"Final child after crossover: {child}")
    
    child = swap_mutation(child)
    logging.info(f"Child after mutation: {child}, fitness: {fitness_function(child)}")