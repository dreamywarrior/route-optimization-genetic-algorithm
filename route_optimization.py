
from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_C_MAX = 1100.0
DEFAULT_T_MAX = 150.0
DEFAULT_POPULATION_SIZE = 120
DEFAULT_GENERATIONS = 500
DEFAULT_CROSSOVER_RATE = 0.80
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_ELITE_SIZE = 2
DEFAULT_SPEED_UNITS_PER_HOUR = 8.0
DEFAULT_RANDOM_CITY_COUNT = 20
DEFAULT_RANDOM_COORD_RANGE = (0.0, 100.0)
PENALTY_WEIGHT = 25.0


@dataclass(frozen=True)
class City:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class ProblemSpec:
    cities: list[City]
    c_max: float = DEFAULT_C_MAX
    t_max: float = DEFAULT_T_MAX


@dataclass
class RouteResult:
    route: list[int]
    distance: float
    time: float
    fitness: float
    feasible: bool
    violation: float


@dataclass
class GAResult:
    best: RouteResult
    best_distances: list[float]
    best_times: list[float]
    checkpoints: dict[int, RouteResult]


class PopulationBuffer:
    """Bounded collection of routes with explicit empty/full errors."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Population capacity must be greater than zero.")
        self.capacity = capacity
        self._items: list[list[int]] = []

    def add(self, route: Sequence[int]) -> None:
        if len(self._items) >= self.capacity:
            raise OverflowError("Population is full; cannot add another route.")
        self._items.append(list(route))

    def extend(self, routes: Iterable[Sequence[int]]) -> None:
        for route in routes:
            self.add(route)

    def to_list(self) -> list[list[int]]:
        return [list(route) for route in self._items]

    def __len__(self) -> int:
        return len(self._items)


def euclidean_distance(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def generate_random_cities(
    count: int = DEFAULT_RANDOM_CITY_COUNT,
    seed: int | None = None,
    coord_range: tuple[float, float] = DEFAULT_RANDOM_COORD_RANGE,
) -> list[City]:
    if count < 2:
        raise ValueError("At least two cities are required.")
    low, high = coord_range
    if high <= low:
        raise ValueError("Invalid coordinate range.")
    rng = random.Random(seed)
    return [City(index=i, x=rng.uniform(low, high), y=rng.uniform(low, high)) for i in range(count)]


def parse_input_file(path: str | Path) -> ProblemSpec:
    """
    Parse assignment-style input files.

    Supported syntax:
      Tmax = 150
      Cmax = 1100
      0 5 10
      1 15 85

    Also supports coordinate-only rows:
      5 10
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    cities: list[City] = []
    c_max = DEFAULT_C_MAX
    t_max = DEFAULT_T_MAX

    for line_no, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        limit_match = re.fullmatch(r"(Tmax|Cmax)\s*=\s*([-+]?\d+(?:\.\d+)?)", line, flags=re.IGNORECASE)
        if limit_match:
            name = limit_match.group(1).lower()
            value = float(limit_match.group(2))
            if value <= 0:
                raise ValueError(f"{limit_match.group(1)} must be positive at line {line_no}.")
            if name == "tmax":
                t_max = value
            else:
                c_max = value
            continue

        parts = [p for p in re.split(r"[,\s]+", line) if p]
        if len(parts) == 2:
            index = len(cities)
            x_text, y_text = parts
        elif len(parts) >= 3:
            try:
                index = int(float(parts[0]))
            except ValueError as exc:
                raise ValueError(f"Invalid city index at line {line_no}: {raw_line!r}") from exc
            x_text, y_text = parts[1], parts[2]
        else:
            raise ValueError(f"Invalid line in input file at line {line_no}: {raw_line!r}")

        try:
            cities.append(City(index=index, x=float(x_text), y=float(y_text)))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value at line {line_no}: {raw_line!r}") from exc

    if len(cities) < 2:
        raise ValueError("At least two cities are required.")

    if any(city.index != idx for idx, city in enumerate(cities)):
        indices = [city.index for city in cities]
        if len(indices) != len(set(indices)):
            raise ValueError("City indices must be unique.")
        cities = sorted(cities, key=lambda city: city.index)

    cities = [City(index=i, x=city.x, y=city.y) for i, city in enumerate(cities)]
    return ProblemSpec(cities=cities, c_max=c_max, t_max=t_max)


def validate_parameters(
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_size: int,
    city_count: int,
    speed_units_per_hour: float,
) -> None:
    if pop_size <= 0:
        raise ValueError("Population size must be greater than zero.")
    if generations <= 0:
        raise ValueError("Generations must be greater than zero.")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("Crossover rate must be between 0 and 1.")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("Mutation rate must be between 0 and 1.")
    if elite_size < 0:
        raise ValueError("Elite size cannot be negative.")
    if elite_size >= pop_size:
        raise ValueError("Elite size must be smaller than population size.")
    if city_count < 2:
        raise ValueError("At least two cities are required.")
    if speed_units_per_hour <= 0:
        raise ValueError("Speed must be positive.")


def validate_route(route: Sequence[int], city_count: int) -> None:
    if len(route) != city_count:
        raise ValueError("Route length must match the number of cities.")
    if route[0] != 0:
        raise ValueError("City 0 must be fixed as the starting city.")
    if set(route) != set(range(city_count)):
        raise ValueError("Route must contain every city exactly once.")


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


def route_violation(distance: float, time: float, c_max: float, t_max: float) -> float:
    cost_violation = max(0.0, distance - c_max) / c_max if c_max > 0 else 0.0
    time_violation = max(0.0, time - t_max) / t_max if t_max > 0 else 0.0
    return cost_violation + time_violation


def evaluate_route(
    route: Sequence[int],
    cities: Sequence[City],
    c_max: float = DEFAULT_C_MAX,
    t_max: float = DEFAULT_T_MAX,
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> RouteResult:
    distance = total_distance(route, cities)
    time = distance / speed_units_per_hour
    violation = route_violation(distance, time, c_max, t_max)
    normalized_distance = distance / c_max if c_max > 0 else distance
    normalized_time = time / t_max if t_max > 0 else time
    fitness = 0.5 * normalized_distance + 0.5 * normalized_time + PENALTY_WEIGHT * violation
    feasible = distance <= c_max and time <= t_max
    return RouteResult(list(route), distance, time, fitness, feasible, violation)


def evaluation_sort_key(result: RouteResult) -> tuple:
    return (0 if result.feasible else 1, result.violation, result.fitness, result.distance, result.time)


def random_route(city_count: int, rng: random.Random) -> list[int]:
    route = list(range(1, city_count))
    rng.shuffle(route)
    return [0, *route]


def repair_route(route: Sequence[int], city_count: int) -> list[int]:
    if city_count < 2:
        raise ValueError("At least two cities are required.")

    repaired = [0]
    used = {0}
    for city in route[1:]:
        if city in used or city < 0 or city >= city_count:
            continue
        repaired.append(city)
        used.add(city)

    for city in range(1, city_count):
        if city not in used:
            repaired.append(city)
            used.add(city)

    return repaired[:city_count]


def rank_selection_from_evaluated(evaluated: Sequence[RouteResult], rng: random.Random) -> list[int]:
    if not evaluated:
        raise ValueError("Population is empty.")
    ranked = sorted(evaluated, key=evaluation_sort_key)
    n = len(ranked)
    total_weight = n * (n + 1) / 2
    pick = rng.uniform(0.0, total_weight)
    cumulative = 0.0

    for idx, result in enumerate(ranked):
        cumulative += (n - idx)
        if pick <= cumulative:
            return list(result.route)
    return list(ranked[0].route)


def ordered_crossover(parent1: Sequence[int], parent2: Sequence[int], rng: random.Random) -> list[int]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")
    city_count = len(parent1)
    if city_count <= 2:
        return list(parent1)

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
    if len(mutated) <= 3:
        return mutated
    i, j = rng.sample(range(1, len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def best_of_routes(routes: Iterable[RouteResult]) -> RouteResult:
    try:
        return min(routes, key=evaluation_sort_key)
    except ValueError as exc:
        raise ValueError("No routes were evaluated.") from exc


def checkpoint_points(generations: int) -> list[int]:
    return sorted({0, min(25, generations), min(100, generations), generations})


def genetic_algorithm(
    cities: Sequence[City],
    pop_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elite_size: int = DEFAULT_ELITE_SIZE,
    seed: int | None = 42,
    c_max: float = DEFAULT_C_MAX,
    t_max: float = DEFAULT_T_MAX,
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> GAResult:
    validate_parameters(
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        city_count=len(cities),
        speed_units_per_hour=speed_units_per_hour,
    )

    rng = random.Random(seed)
    population = PopulationBuffer(pop_size)
    population.extend(random_route(len(cities), rng) for _ in range(pop_size))

    checkpoints_needed = checkpoint_points(generations)
    checkpoint_map: dict[int, RouteResult] = {}
    best_distances: list[float] = []
    best_times: list[float] = []

    evaluated = [
        evaluate_route(route, cities, c_max, t_max, speed_units_per_hour)
        for route in population.to_list()
    ]
    current_best = best_of_routes(evaluated)
    global_best = current_best
    best_distances.append(current_best.distance)
    best_times.append(current_best.time)
    if 0 in checkpoints_needed:
        checkpoint_map[0] = current_best

    for generation in range(1, generations + 1):
        ranked = sorted(evaluated, key=evaluation_sort_key)
        next_population = PopulationBuffer(pop_size)
        next_population.extend(result.route for result in ranked[:elite_size])

        while len(next_population) < pop_size:
            parent1 = rank_selection_from_evaluated(evaluated, rng)
            parent2 = rank_selection_from_evaluated(evaluated, rng)
            child = list(parent1)
            if rng.random() < crossover_rate:
                child = ordered_crossover(parent1, parent2, rng)
            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng)
            next_population.add(repair_route(child, len(cities)))

        population = next_population
        evaluated = [
            evaluate_route(route, cities, c_max, t_max, speed_units_per_hour)
            for route in population.to_list()
        ]
        current_best = best_of_routes(evaluated)
        best_distances.append(current_best.distance)
        best_times.append(current_best.time)

        if evaluation_sort_key(current_best) < evaluation_sort_key(global_best):
            global_best = current_best
        if generation in checkpoints_needed:
            checkpoint_map[generation] = current_best

    resolved_checkpoints: dict[int, RouteResult] = {}
    last_seen: RouteResult | None = None
    for point in checkpoints_needed:
        if point in checkpoint_map:
            last_seen = checkpoint_map[point]
            resolved_checkpoints[point] = checkpoint_map[point]
        elif last_seen is not None:
            resolved_checkpoints[point] = last_seen
        else:
            resolved_checkpoints[point] = global_best

    return GAResult(global_best, best_distances, best_times, resolved_checkpoints)


def random_search(
    cities: Sequence[City],
    evaluations: int,
    seed: int | None = 100,
    c_max: float = DEFAULT_C_MAX,
    t_max: float = DEFAULT_T_MAX,
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> RouteResult:
    if evaluations <= 0:
        raise ValueError("Evaluations must be greater than zero.")
    rng = random.Random(seed)
    best = evaluate_route(random_route(len(cities), rng), cities, c_max, t_max, speed_units_per_hour)
    for _ in range(evaluations - 1):
        candidate = evaluate_route(random_route(len(cities), rng), cities, c_max, t_max, speed_units_per_hour)
        if evaluation_sort_key(candidate) < evaluation_sort_key(best):
            best = candidate
    return best


def hill_climbing(
    cities: Sequence[City],
    evaluations: int,
    restarts: int = 20,
    seed: int | None = 200,
    c_max: float = DEFAULT_C_MAX,
    t_max: float = DEFAULT_T_MAX,
    speed_units_per_hour: float = DEFAULT_SPEED_UNITS_PER_HOUR,
) -> RouteResult:
    if evaluations <= 0:
        raise ValueError("Evaluations must be greater than zero.")
    if restarts <= 0:
        raise ValueError("Restarts must be greater than zero.")
    rng = random.Random(seed)
    best_overall: RouteResult | None = None
    evaluations_per_restart = max(1, evaluations // restarts)

    for _ in range(restarts):
        current = evaluate_route(random_route(len(cities), rng), cities, c_max, t_max, speed_units_per_hour)
        for _ in range(evaluations_per_restart - 1):
            neighbour = evaluate_route(
                swap_mutation(current.route, rng), cities, c_max, t_max, speed_units_per_hour
            )
            if evaluation_sort_key(neighbour) < evaluation_sort_key(current):
                current = neighbour
        if best_overall is None or evaluation_sort_key(current) < evaluation_sort_key(best_overall):
            best_overall = current

    if best_overall is None:
        raise RuntimeError("Hill climbing did not evaluate any routes.")
    return best_overall


def comparison_table(results: dict[str, RouteResult]) -> str:
    header = [
        "Method                Distance       Time     Fitness     Feasible",
        "-" * 68,
    ]
    rows = [
        f"{method:<20} {result.distance:10.2f} {result.time:10.2f} {result.fitness:10.4f}   {'Yes' if result.feasible else 'No'}"
        for method, result in results.items()
    ]
    return "\n".join(header + rows)


def build_constraint_status(result: RouteResult, c_max: float, t_max: float) -> str:
    time_status = "SATISFIED" if result.time <= t_max else "NOT SATISFIED"
    cost_status = "SATISFIED" if result.distance <= c_max else "NOT SATISFIED"
    return f"Time <= Tmax : {time_status}\nCost <= Cmax : {cost_status}"


def apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(
            context="notebook",
            style="whitegrid",
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
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")


def route_step_map(route: Sequence[int]) -> dict[int, int]:
    return {city: step + 1 for step, city in enumerate(route)}


def draw_route_on_axis(ax, cities: Sequence[City], result: RouteResult, title: str) -> None:
    route = result.route
    ordered = list(route) + [route[0]]
    step_map = route_step_map(route)

    ax.scatter(
        [city.x for city in cities],
        [city.y for city in cities],
        s=70,
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
        label="Cities",
    )

    start_city = cities[0]
    ax.scatter([start_city.x], [start_city.y], s=220, marker="*", zorder=6, label="Start City (0)")

    for city in cities:
        if city.index == 0:
            ax.annotate(
                "0\n(start)",
                (city.x, city.y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                weight="bold",
            )
        else:
            ax.annotate(
                f"{city.index}\n#{step_map.get(city.index, '')}",
                (city.x, city.y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    xs = [cities[i].x for i in ordered]
    ys = [cities[i].y for i in ordered]
    ax.plot(xs, ys, linewidth=2.0, zorder=2, label="Route")
    for start, end in zip(ordered, ordered[1:]):
        ax.annotate(
            "",
            xy=(cities[end].x, cities[end].y),
            xytext=(cities[start].x, cities[start].y),
            arrowprops={"arrowstyle": "->", "lw": 0.95, "alpha": 0.8},
        )

    ax.plot(
        [cities[route[-1]].x, cities[0].x],
        [cities[route[-1]].y, cities[0].y],
        linestyle="--",
        linewidth=2.2,
        label="Return edge",
    )

    ax.set_title(f"{title}\nDistance: {result.distance:.2f} | Time: {result.time:.2f}", pad=12)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="best")


def plot_best_route(cities: Sequence[City], result: RouteResult, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    draw_route_on_axis(ax, cities, result, "Best Route Found by Genetic Algorithm")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_route_evolution_grid(
    cities: Sequence[City],
    checkpoints: dict[int, RouteResult],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if not checkpoints:
        raise ValueError("No checkpoint routes available for plotting.")

    apply_plot_style()
    points = sorted(checkpoints.items(), key=lambda item: item[0])
    while len(points) < 4:
        points.append(points[-1])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    panel_titles = [
        "Initial Traversal",
        "Traversal After Generation 25",
        "Traversal After Generation 100",
        "Final Traversal",
    ]

    for ax, (generation, result), panel_title in zip(axes, points[:4], panel_titles):
        draw_route_on_axis(ax, cities, result, f"{panel_title} (Generation {generation})")
        ax.text(
            0.02,
            0.02,
            f"Generation {generation}\nCost: {result.distance:.2f}\nTime: {result.time:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
        )

    fig.suptitle("Route Evolution Across Generations", fontsize=18, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(values: Sequence[float], ylabel: str, title: str, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    if not values:
        raise ValueError("No convergence values to plot.")
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(range(len(values)), values, linewidth=2.2)
    ax.scatter([0, len(values) - 1], [values[0], values[-1]], s=50, zorder=3)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.text(len(values) - 1, values[-1], f"  Final: {values[-1]:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def plot_algorithm_comparison(
    results: dict[str, RouteResult],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()

    methods = list(results.keys())
    distances = [results[m].distance for m in methods]
    times = [results[m].time for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        distances,
        width,
        label="Distance",
    )

    bars2 = ax.bar(
        x + width / 2,
        times,
        width,
        label="Time",
    )

    ax.set_title(
        "Algorithm Performance Comparison",
        fontsize=15,
        weight="bold",
        pad=12,
    )

    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def summarize_convergence(values: Sequence[float]) -> str:
    if not values:
        return "No convergence data available."
    return f"improved from {values[0]:.2f} -> {values[-1]:.2f}"


def write_execution_log_text(
    output_path: str | Path,
    results: dict[str, RouteResult],
    ga_result: GAResult,
    c_max: float,
    t_max: float,
) -> None:
    best_method = min(results.items(), key=lambda item: evaluation_sort_key(item[1]))[0]
    lines = [
        "PS15 Route Optimization - Execution Log",
        "",
        "Problem Formulation:",
        "Permutation-based route optimization with constraints on total distance and travel time.",
        "",
        "Representation:",
        "Permutation encoding with city 0 fixed as the start city.",
        "",
        "Operators Used:",
        "Ordered Crossover (OX), Swap Mutation, and Rank Selection.",
        "",
        "Constraint Handling:",
        f"Cmax = {c_max}, Tmax = {t_max}. Feasible routes are always preferred over infeasible routes.",
        "",
        "Alternative Modeling Approach:",
        "A Pareto-based approach such as NSGA-II could optimize distance and time separately.",
        "",
        "Results Summary:",
        comparison_table(results),
        "",
        f"Best observed method: {best_method}",
        f"GA distance convergence: {summarize_convergence(ga_result.best_distances)}",
        f"GA time convergence: {summarize_convergence(ga_result.best_times)}",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_output(
    output_path: str | Path,
    ga_result: GAResult,
    results: dict[str, RouteResult],
    c_max: float,
    t_max: float,
    pop_size: int,
    generations: int,
) -> None:
    best = ga_result.best
    route_text = " -> ".join(map(str, [*best.route, best.route[0]]))
    content = [
        "Genetic Algorithm Results",
        "-------------------------",
        f"Best Route: {route_text}",
        f"Total Distance (Cost): {best.distance:.2f}",
        f"Total Travel Time: {best.time:.2f} hours",
        "Constraints Status:",
        build_constraint_status(best, c_max, t_max),
        "",
        "GA Parameters:",
        f"Population Size = {pop_size}",
        f"Generations = {generations}",
        "Crossover = Ordered Crossover (OX)",
        "Mutation = Swap Mutation",
        "Selection = Rank Selection",
        "",
        "Performance Comparison",
        "---------------------------------------------------",
        comparison_table(results),
        "",
        "Convergence:",
        f"Best Distance {summarize_convergence(ga_result.best_distances)}",
        f"Best Time {summarize_convergence(ga_result.best_times)}",
        "",
    ]
    Path(output_path).write_text("\n".join(content), encoding="utf-8")


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_problem(
    input_path: str | Path | None,
    allow_random_fallback: bool,
    random_city_count: int,
    seed: int | None,
) -> ProblemSpec:
    if random_city_count < 2:
        raise ValueError("Random city count must be at least two.")

    if input_path is None:
        print(f"No input file provided. Generating {random_city_count} random cities instead.")
        return ProblemSpec(cities=generate_random_cities(random_city_count, seed=seed))

    file_path = Path(input_path)
    if file_path.exists():
        return parse_input_file(file_path)

    if allow_random_fallback:
        print(
            f"Warning: input file '{file_path}' was not found. "
            f"Generating {random_city_count} random cities instead."
        )
        return ProblemSpec(cities=generate_random_cities(random_city_count, seed=seed))

    raise FileNotFoundError(f"Input file not found: {file_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    problem = load_problem(args.input, args.random_fallback, args.random_city_count, args.seed)
    cities = problem.cities
    c_max = args.c_max if args.c_max is not None else problem.c_max
    t_max = args.t_max if args.t_max is not None else problem.t_max

    validate_parameters(
        pop_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        city_count=len(cities),
        speed_units_per_hour=args.speed,
    )

    output_dir = ensure_output_dir(args.output_dir)
    evaluations = args.population * args.generations

    ga_result = genetic_algorithm(
        cities=cities,
        pop_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        seed=args.seed,
        c_max=c_max,
        t_max=t_max,
        speed_units_per_hour=args.speed,
    )
    random_result = random_search(
        cities=cities,
        evaluations=evaluations,
        seed=args.seed + 1,
        c_max=c_max,
        t_max=t_max,
        speed_units_per_hour=args.speed,
    )
    hill_result = hill_climbing(
        cities=cities,
        evaluations=evaluations,
        restarts=args.restarts,
        seed=args.seed + 2,
        c_max=c_max,
        t_max=t_max,
        speed_units_per_hour=args.speed,
    )

    results = {
        "Genetic Algorithm": ga_result.best,
        "Random Search": random_result,
        "Hill Climbing": hill_result,
    }

    write_output(args.output, ga_result, results, c_max, t_max, args.population, args.generations)
    write_execution_log_text(output_dir / "execution_log.txt", results, ga_result, c_max, t_max)
    plot_best_route(cities, ga_result.best, output_dir / "best_route.png")
    plot_route_evolution_grid(cities, ga_result.checkpoints, output_dir / "route_evolution_grid.png")
    plot_convergence(
        ga_result.best_distances,
        "Best Distance",
        "Distance Convergence",
        output_dir / "distance_convergence.png",
    )
    plot_convergence(
        ga_result.best_times,
        "Best Time",
        "Time Convergence",
        output_dir / "time_convergence.png",
    )
    plot_algorithm_comparison(
        results,
        output_dir / "algorithm_comparison.png",
    )

    print("Genetic Algorithm Results")
    print("-------------------------")
    print(f"Best Route: {' -> '.join(map(str, [*ga_result.best.route, ga_result.best.route[0]]))}")
    print(f"Total Distance (Cost): {ga_result.best.distance:.2f}")
    print(f"Total Travel Time: {ga_result.best.time:.2f} hours")
    print("Constraints Status:")
    print(build_constraint_status(ga_result.best, c_max, t_max))
    print()
    print("GA Parameters:")
    print(f"Population Size = {args.population}")
    print(f"Generations = {args.generations}")
    print("Crossover = Ordered Crossover (OX)")
    print("Mutation = Swap Mutation")
    print("Selection = Rank Selection")
    print()
    print("Performance Comparison")
    print("---------------------------------------------------")
    print(comparison_table(results))
    print()
    print("Convergence for Genetic Algorithm:")
    print(f"Best Distance {summarize_convergence(ga_result.best_distances)}")
    print(f"Best Time {summarize_convergence(ga_result.best_times)}")
    print(f"\nWrote {args.output}, plots, and execution log to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PS15 route optimization using a genetic algorithm.")
    parser.add_argument("--input", default="inputPS15.txt", help="Path to the assignment input file.")
    parser.add_argument("--output", default="outputPS15.txt", help="Path to the assignment output file.")
    parser.add_argument("--output-dir", default="results", help="Directory for plots and helper logs.")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION_SIZE, help="Population size.")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="Number of generations.")
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE, help="Crossover probability.")
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE, help="Mutation probability.")
    parser.add_argument("--elite-size", type=int, default=DEFAULT_ELITE_SIZE, help="Number of elites to preserve.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_UNITS_PER_HOUR, help="Travel speed units per hour.")
    parser.add_argument("--c-max", type=float, default=None, help="Override Cmax from file.")
    parser.add_argument("--t-max", type=float, default=None, help="Override Tmax from file.")
    parser.add_argument("--restarts", type=int, default=20, help="Number of hill-climbing restarts.")
    parser.add_argument(
        "--no-random-fallback",
        dest="random_fallback",
        action="store_false",
        help="Do not generate random cities if the input file is missing.",
    )
    parser.set_defaults(random_fallback=True)
    parser.add_argument(
        "--random-city-count",
        type=int,
        default=DEFAULT_RANDOM_CITY_COUNT,
        help="Number of randomly generated cities when fallback is used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_pipeline(args)
    except (FileNotFoundError, ValueError, OverflowError, IndexError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
