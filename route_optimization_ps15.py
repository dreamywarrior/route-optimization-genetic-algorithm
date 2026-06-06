"""
=============================================================
BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
WORK INTEGRATED LEARNING PROGRAMMES DIVISION

Artificial and Computational Intelligence - Assignment 1
Part #5: Route Optimization using Genetic Algorithm
Team Number: 191

Tasks Implemented:
  Task 4 - Random Search + Hill Climbing + Comparison Table
  Task 5 - Seaborn Plots + Design Document Notes + Final Submission Helpers

Group Number (G): 191
=============================================================
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


C_MAX = 1100.0
T_MAX = 150.0
DEFAULT_POP_SIZE = 120
DEFAULT_GENERATIONS = 500
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_ELITE_SIZE = 2
DEFAULT_SPEED_UNITS_PER_HOUR = 8.0


@dataclass(frozen=True)
class City:
    index: int
    x: float
    y: float


@dataclass
class RouteResult:
    route: list[int]
    distance: float
    time: float
    fitness: float
    feasible: bool


@dataclass
class GAResult:
    best: RouteResult
    best_distances: list[float]
    best_times: list[float]


def read_cities(input_path: str | Path) -> list[City]:
    """Read cities from inputPS15.txt-style files without hardcoding values."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    cities: list[City] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()
        if len(parts) == 2:
            index = len(cities)
            x_text, y_text = parts
        elif len(parts) >= 3:
            index = int(parts[0])
            x_text, y_text = parts[1], parts[2]
        else:
            raise ValueError(f"Invalid city row at line {line_no}: {raw_line!r}")

        try:
            cities.append(City(index=index, x=float(x_text), y=float(y_text)))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value at line {line_no}: {raw_line!r}") from exc

    if len(cities) < 2:
        raise ValueError("At least two cities are required.")

    seen = {city.index for city in cities}
    if len(seen) != len(cities):
        raise ValueError("City indices must be unique.")

    return cities


def euclidean_distance(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def validate_route(route: Sequence[int], city_count: int) -> None:
    expected = set(range(city_count))
    actual = set(route)
    if len(route) != city_count or actual != expected:
        raise ValueError("Route must contain every city exactly once.")
    if route[0] != 0:
        raise ValueError("City 0 must be fixed as the route start.")


def total_distance(route: Sequence[int], cities: Sequence[City]) -> float:
    validate_route(route, len(cities))
    distance = 0.0
    for current, nxt in zip(route, route[1:]):
        distance += euclidean_distance(cities[current], cities[nxt])
    distance += euclidean_distance(cities[route[-1]], cities[route[0]])
    return distance


def total_time(
    route: Sequence[int],
    cities: Sequence[City],
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> float:
    if speed_units_per_hour <= 0:
        raise ValueError("Speed must be positive.")
    return total_distance(route, cities) / speed_units_per_hour


def evaluate_route(
    route: Sequence[int],
    cities: Sequence[City],
    c_max: float = C_MAX,
    t_max: float = T_MAX,
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> RouteResult:
    distance = total_distance(route, cities)
    time = distance / speed_units_per_hour
    normalized_distance = distance / c_max
    normalized_time = time / t_max
    penalty = max(0.0, normalized_distance - 1.0) + max(0.0, normalized_time - 1.0)
    fitness = 0.5 * normalized_distance + 0.5 * normalized_time + 10.0 * penalty
    return RouteResult(
        route=list(route),
        distance=distance,
        time=time,
        fitness=fitness,
        feasible=distance <= c_max and time <= t_max,
    )


def random_route(city_count: int, rng: random.Random) -> list[int]:
    route = list(range(1, city_count))
    rng.shuffle(route)
    return [0, *route]


def rank_selection(population: Sequence[list[int]], cities: Sequence[City], rng: random.Random) -> list[int]:
    ranked = sorted(population, key=lambda route: evaluate_route(route, cities).fitness, reverse=True)
    rank_total = len(ranked) * (len(ranked) + 1) / 2
    pick = rng.random()
    cumulative = 0.0

    for rank, route in enumerate(ranked, start=1):
        cumulative += rank / rank_total
        if pick <= cumulative:
            return list(route)
    return list(ranked[-1])


def ordered_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> list[int]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")

    city_count = len(parent1)
    start, end = sorted(rng.sample(range(1, city_count), 2))
    child: list[int | None] = [None] * city_count
    child[0] = 0
    child[start:end] = parent1[start:end]

    fill_values = [city for city in parent2[1:] if city not in child]
    fill_index = 0
    for i in list(range(1, start)) + list(range(end, city_count)):
        child[i] = fill_values[fill_index]
        fill_index += 1

    completed = [int(city) for city in child]
    validate_route(completed, city_count)
    return completed


def swap_mutation(route: Sequence[int], rng: random.Random) -> list[int]:
    mutated = list(route)
    i, j = rng.sample(range(1, len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def genetic_algorithm(
    cities: Sequence[City],
    pop_size: int = DEFAULT_POP_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elite_size: int = DEFAULT_ELITE_SIZE,
    seed: int | None = 42,
) -> GAResult:
    rng = random.Random(seed)
    population = [random_route(len(cities), rng) for _ in range(pop_size)]
    best_distances: list[float] = []
    best_times: list[float] = []

    for _ in range(generations + 1):
        evaluated = sorted((evaluate_route(route, cities) for route in population), key=lambda item: item.fitness)
        best_distances.append(evaluated[0].distance)
        best_times.append(evaluated[0].time)

        next_population = [result.route for result in evaluated[:elite_size]]
        while len(next_population) < pop_size:
            parent1 = rank_selection(population, cities, rng)
            parent2 = rank_selection(population, cities, rng)
            child = ordered_crossover(parent1, parent2, rng) if rng.random() < crossover_rate else list(parent1)
            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng)
            next_population.append(child)

        population = next_population

    best = min((evaluate_route(route, cities) for route in population), key=lambda item: item.fitness)
    return GAResult(best=best, best_distances=best_distances, best_times=best_times)


def random_search(cities: Sequence[City], evaluations: int, seed: int | None = 100) -> RouteResult:
    rng = random.Random(seed)
    best = evaluate_route(random_route(len(cities), rng), cities)
    for _ in range(evaluations - 1):
        candidate = evaluate_route(random_route(len(cities), rng), cities)
        if candidate.fitness < best.fitness:
            best = candidate
    return best


def hill_climbing(
    cities: Sequence[City],
    evaluations: int,
    restarts: int = 20,
    seed: int | None = 200,
) -> RouteResult:
    rng = random.Random(seed)
    best_overall: RouteResult | None = None
    evaluations_per_restart = max(1, evaluations // restarts)

    for _ in range(restarts):
        current_route = random_route(len(cities), rng)
        current = evaluate_route(current_route, cities)

        for _ in range(evaluations_per_restart - 1):
            neighbour = evaluate_route(swap_mutation(current.route, rng), cities)
            if neighbour.fitness < current.fitness:
                current = neighbour

        if best_overall is None or current.fitness < best_overall.fitness:
            best_overall = current

    if best_overall is None:
        raise RuntimeError("Hill climbing did not evaluate any routes.")
    return best_overall


def comparison_table(results: dict[str, RouteResult]) -> str:
    rows = ["Method               Distance       Time     Fitness   Feasible"]
    rows.append("-" * 62)
    for method, result in results.items():
        rows.append(
            f"{method:<18} {result.distance:10.2f} {result.time:10.2f} "
            f"{result.fitness:10.4f}   {'Yes' if result.feasible else 'No'}"
        )
    return "\n".join(rows)


def apply_plot_style() -> None:
    """Apply a polished seaborn theme when available, with a matplotlib fallback."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(
            context="notebook",
            style="whitegrid",
            palette="deep",
            rc={
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "axes.titlesize": 15,
                "axes.labelsize": 12,
                "axes.titleweight": "bold",
                "grid.alpha": 0.25,
                "legend.frameon": False,
            },
        )
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")


def plot_best_route(cities: Sequence[City], route: Sequence[int], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    ordered = list(route) + [route[0]]
    xs = [cities[i].x for i in ordered]
    ys = [cities[i].y for i in ordered]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        [city.x for city in cities],
        [city.y for city in cities],
        s=85,
        color="#2563EB",
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        label="Cities",
    )
    for city in cities:
        ax.text(city.x, city.y, f" {city.index}", fontsize=9, weight="bold", color="#111827")
    ax.plot(xs, ys, color="#DC2626", linewidth=2.0, alpha=0.9, zorder=2, label="Best route")
    for start, end in zip(ordered, ordered[1:]):
        ax.annotate(
            "",
            xy=(cities[end].x, cities[end].y),
            xytext=(cities[start].x, cities[start].y),
            arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.1, "alpha": 0.75},
        )
    ax.set_title("Best Route Found by Genetic Algorithm", pad=14)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(values: Sequence[float], ylabel: str, title: str, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(len(values)), values, color="#059669", linewidth=2.3)
    ax.fill_between(range(len(values)), values, min(values), color="#10B981", alpha=0.12)
    ax.scatter([len(values) - 1], [values[-1]], color="#047857", s=60, zorder=4)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.text(
        len(values) - 1,
        values[-1],
        f"  Final: {values[-1]:.2f}",
        va="center",
        fontsize=10,
        color="#064E3B",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_design_document(results: dict[str, RouteResult], output_path: str | Path) -> None:
    best_method = min(results.items(), key=lambda item: item[1].fitness)[0]
    lines = [
        "# ACI Assignment 1 - PS15 Design Document",
        "",
        "## Problem Formulation",
        "The problem is modeled as a constrained travelling-salesperson route optimization task. "
        "Each route visits every city exactly once, returns to city 0, and is evaluated by total "
        "Euclidean travel distance and travel time while respecting Cmax = 1100 and Tmax = 150.",
        "",
        "## Algorithm Design and Justification",
        "The genetic algorithm uses permutation encoding with city 0 fixed as the start/end point. "
        "Rank selection reduces domination by early strong candidates, ordered crossover preserves "
        "valid route subsequences, swap mutation keeps diversity, and elitism protects the best "
        "routes from being lost between generations.",
        "",
        "## Alternate Modeling Approach",
        "An alternate approach is Pareto-based NSGA-II, where distance and time are optimized as "
        "separate objectives instead of being combined into one weighted fitness score. Another "
        "lighter alternative is Nearest Neighbour followed by 2-opt local improvement. NSGA-II gives "
        "a richer feasible frontier, while Nearest Neighbour + 2-opt is faster but more prone to "
        "local optima.",
        "",
        "## Results and Analysis",
        comparison_table(results),
        "",
        f"The best fitness in this run was produced by {best_method}. The GA is expected to "
        "outperform pure random search because it reuses strong partial routes across generations. "
        "It is also more robust than hill climbing because crossover and mutation explore multiple "
        "regions of the search space instead of accepting only local improvements from one route.",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_output(output_path: str | Path, ga_result: GAResult, results: dict[str, RouteResult]) -> None:
    best = ga_result.best
    route_text = " -> ".join(map(str, [*best.route, best.route[0]]))
    content = [
        "ACI Assignment 1 - PS15 Route Optimization Output",
        "",
        f"Best Route: {route_text}",
        f"Best Distance: {best.distance:.2f}",
        f"Best Time: {best.time:.2f}",
        f"Best Fitness: {best.fitness:.4f}",
        f"Constraint Status: {'Feasible' if best.feasible else 'Infeasible'}",
        "",
        "Performance Comparison",
        comparison_table(results),
        "",
    ]
    Path(output_path).write_text("\n".join(content), encoding="utf-8")


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(args: argparse.Namespace) -> None:
    cities = read_cities(args.input)
    output_dir = ensure_output_dir(args.output_dir)
    evaluations = args.population * args.generations

    ga_result = genetic_algorithm(
        cities,
        pop_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        seed=args.seed,
    )
    random_result = random_search(cities, evaluations=evaluations, seed=args.seed + 1)
    hill_result = hill_climbing(cities, evaluations=evaluations, seed=args.seed + 2)

    results = {
        "Genetic Algorithm": ga_result.best,
        "Random Search": random_result,
        "Hill Climbing": hill_result,
    }

    write_output(args.output, ga_result, results)
    plot_best_route(cities, ga_result.best.route, output_dir / "best_route.png")
    plot_convergence(
        ga_result.best_distances,
        "Best distance",
        "Distance Convergence",
        output_dir / "distance_convergence.png",
    )
    plot_convergence(
        ga_result.best_times,
        "Best travel time",
        "Time Convergence",
        output_dir / "time_convergence.png",
    )
    write_design_document(results, output_dir / "design_document.md")

    print(comparison_table(results))
    print(f"\nWrote {args.output} and plots/design notes to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PS15 route optimization using a genetic algorithm.")
    parser.add_argument("--input", default="inputPS15.txt", help="Path to input city coordinate file.")
    parser.add_argument("--output", default="outputPS15.txt", help="Path to assignment output file.")
    parser.add_argument("--output-dir", default="results", help="Directory for plots and design notes.")
    parser.add_argument("--population", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    parser.add_argument("--elite-size", type=int, default=DEFAULT_ELITE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
