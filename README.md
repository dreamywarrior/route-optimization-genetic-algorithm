# Route Optimization — Genetic Algorithm

A compact Python project implementing a genetic algorithm (GA) for the PS15 route optimization assignment. The GA searches for a permutation-based route visiting all cities exactly once with city 0 fixed as the start, while respecting cost (distance) and time constraints.

**Key Features**
- Permutation representation with city 0 fixed as the start.
- Ordered Crossover (OX), Swap Mutation, and Rank Selection.
- Constraint handling with penalty-based evaluation (distance and time vs Cmax/Tmax).
- Comparison with Random Search and Hill Climbing baselines.
- Plots for route, evolution across generations, and convergence.

**Repository**
- Main script: [route_optimization.py](route_optimization.py)
- Example input: `inputPS15.txt` (assignment-style format)
- Outputs and plots: `results/`

**Requirements**
- Python 3.10+ recommended
- See `requirements.txt` for exact dependencies. Typical packages:
	- `numpy`
	- `matplotlib`
	- `seaborn` (optional, falls back to matplotlib style)

**Input format**
The tool accepts assignment-style text files. Supported lines:
- Limits: `Tmax = 150` or `Cmax = 1100`
- City rows (index x y) or (x y) — if indices are omitted they are inferred sequentially.

Example minimal input file:
```
Tmax = 150
Cmax = 1100

0 5 10
1 15 85
5 10
```

**Quickstart**
Run with defaults (reads `inputPS15.txt`, writes `outputPS15.txt`, and populates `results/`):

```bash
python route_optimization.py
```

Change population, generations, or seed:

```bash
python route_optimization.py --population 150 --generations 800 --seed 123
```

Disable random fallback if input file is missing:

```bash
python route_optimization.py --no-random-fallback
```

Override constraints:

```bash
python route_optimization.py --c-max 1200 --t-max 140
```

**Outputs**
- `outputPS15.txt`: summary of the GA results and comparison table.
- `results/` folder (default):
	- `best_route.png`
	- `route_evolution_grid.png`
	- `distance_convergence.png`
	- `time_convergence.png`
	- `algorithm_comparison.png`
	- `execution_log.txt`

**Main parameters (CLI)**
- `--population`: population size (default 120)
- `--generations`: number of generations (default 500)
- `--crossover-rate`, `--mutation-rate`: operator probabilities
- `--elite-size`: number of elites preserved each generation
- `--speed`: travel speed units per hour used to compute time
- `--random-city-count`: number of cities when generating random fallback

**Development notes**
- Key algorithms are implemented in `route_optimization.py` (functions such as `genetic_algorithm`, `random_search`, and `hill_climbing`).
- The code includes input parsing, validation, plotting helpers, and a simple logging output.
- Tests or additional automation are not included; feel free to add unit tests for evaluation and operators.

