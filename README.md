# Route Optimization - Genetic Algorithm (PS15)

Multi-objective route optimization using a permutation-based GA.  
Given 20 cities, find a route visiting every city exactly once and returning to the origin while minimizing total distance and travel time.

---

## What's done (as on 04/06)

### Part 2 — Rank Selection (replaces Tournament Selection)

The original code used **tournament selection** — randomly pick 5 individuals, sort them, run roulette wheel on those 5. The assignment requires **rank selection**.

**What was changed:**

- Removed `roulette_wheel_selection()` and `tournament_selection()`
- Removed `TOURNAMENT_SIZE` and `WINNER_PER_TOURNAMENT` constants
- Added `build_rank_probabilities(pop_size)` — sorts the entire population by distance, assigns rank 1 to the worst and rank N to the best. Computes a cumulative probability array once per generation so we don't redo the math for every parent pair.
- Added `rank_selection(sorted_population, cumulative)` — generates a random number in [0,1) and walks the cumulative array to pick a parent. Repeats for a second parent (no duplicates). Better-ranked individuals have proportionally higher chance of being selected.

### Part 3 — Main GA Loop + Elitism + Convergence Tracking

The original `__main__` ran **one** selection + crossover + mutation and stopped. The `GENERATIONS = 500` constant was never used.

**What was changed:**

- Added `ELITISM_COUNT = 2` parameter
- Added a 500-generation loop where each generation:
  1. Sorts population by distance (descending — worst first, best last)
  2. Builds the rank probability wheel once
  3. Preserves the top `ELITISM_COUNT` individuals unchanged into the next generation (elitism)
  4. Fills remaining 118 slots via rank selection → ordered crossover → swap mutation
  5. Replaces old population with new one
- Added `best_distances[]` and `best_times[]` lists that record the best value at each generation (convergence data)
- Added progress printing every 50 generations
- Added final results summary (best route, distance, time, convergence range)

---

## What was NOT touched

These functions are **identical** to the original `GA_Operations.py` in `Version-1` branch:

* `is_valid_route()`
* `generate_city_coordinates()`
* `generate_route()`
* `populate_population()`
* `fitness_function()`
* `ordered_crossover()`
* `swap_mutation()`|

---

## What's pending

### Part 1 — Problem Setup + Fitness Function + File I/O

- [ ] Read city coordinates from `inputPS15.txt` instead of generating randomly (evaluator will use their own input file — hardcoded values will fail)
- [ ] Read `Tmax` and `Cmax` from the input file
- [ ] Add constraint handling in fitness — penalize routes where time > Tmax or distance > Cmax
- [ ] Make fitness return a **scalar value** (currently returns `[dist, time]` list) that accounts for both distance and time (multi-objective trade-off)
- [ ] Write results to `outputPS15.txt` in the sample format (route with →, constraints status, GA parameters, comparison table, convergence summary)

### Part 4 — Random Search + Hill Climbing + Comparison Table

- [ ] Implement random search baseline (generate N random routes, keep best)
- [ ] Implement hill climbing baseline (swap-based local search with random restarts)
- [ ] Generate the performance comparison table: Method | Distance | Time | Fitness

### Part 5 — Plots + Design Doc + Final Submission

- [ ] Best route visualization (2D scatter with route lines)
- [ ] Distance convergence graph (distance vs generation)
- [ ] Time convergence graph (time vs generation)
- [ ] `designPS15_<group_id>.pdf` — 4-page design document with one alternate modeling approach
- [ ] `[Group id]_Contribution.xlsx` — member contribution percentages
- [ ] Zip everything as `[Group id]_A1_PS15_XXXXXXXXXX.zip`

---

## How to run

```bash
python GA_Operations.py
```

Output is printed to console. Logs are written to `logs/GA_Operations_<timestamp>.log`.

---
