from __future__ import annotations

import random
import statistics
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from parser.abz_parser import parse_all_abz
from trajectory.ils import ILSParams, iterated_local_search
from population.gs import Population


# -----------------------
# Pick your defaults here
# -----------------------

DEFAULT_ILS_CFG = dict(
    iterations=87,
    local_steps=800, #low for fast
    swaps=2,
    accept_worse_prob=0.16,
    init="order",
    verbose_every=0,
)

DEFAULT_GA_CFG = dict(
    population_size=175,
    mutation_rate=0.005,
    crossover_rate=0.055,
    new_genomes_per_gen=150,
    max_iterations_without_improvement=50,
    verbose_every=0,
)


# -----------------------
# Helpers
# -----------------------

def summarize(scores: List[float], runtime_sec: float) -> Dict[str, Any]:
    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "runtime_sec": runtime_sec,
    }


def pick_instance(instances, *, idx: Optional[int] = None, name: Optional[str] = None):
    if idx is not None:
        return instances[idx]
    if name is not None:
        for inst in instances:
            if inst.name == name:
                return inst
        raise ValueError(f"Instance name not found: {name}")
    return instances[0]


# -----------------------
# Solver runners
# -----------------------

def run_ils(inst, cfg: Dict[str, Any], seed: int):
    params = ILSParams(
        iterations=cfg["iterations"],
        local_steps=cfg["local_steps"],
        swaps=cfg["swaps"],
        seed=seed,
        accept_worse_prob=cfg["accept_worse_prob"],
        verbose_every=cfg.get("verbose_every", 0),
        init=cfg["init"],
    )
    res = iterated_local_search(inst, params)
    return res.best_makespan, params


def run_ga(inst, cfg: Dict[str, Any], seed: int):
    random.seed(seed)

    pop = Population(
        problem_instance=inst,
        population_size=cfg["population_size"],
        mutation_rate=cfg["mutation_rate"],
        crossover_rate=cfg.get("crossover_rate", 0.5),
        max_iterations_without_improvement=cfg["max_iterations_without_improvement"],
        new_genomes_per_gen=cfg["new_genomes_per_gen"],
    )

    # If your Population.simulation supports verbose_every, pass it.
    # Otherwise, remove verbose_every.
    try:
        best = pop.simulation(verbose_every=cfg.get("verbose_every", 0))
    except TypeError:
        best = pop.simulation()

    return best, cfg


def compare_on_instance(
    inst,
    *,
    ils_cfg: Dict[str, Any],
    ga_cfg: Dict[str, Any],
    repeats: int = 10,
    base_seed: int = 42,
):
    # generate deterministic seeds for fairness
    seeds = [base_seed + i for i in range(repeats)]

    # ---- ILS ----
    ils_scores: List[int] = []
    ils_best = (float("inf"), None, None)  # (score, seed, params)
    t0 = time.perf_counter()
    for sd in seeds:
        score, params = run_ils(inst, ils_cfg, sd)
        ils_scores.append(score)
        if score < ils_best[0]:
            ils_best = (score, sd, params)
    t1 = time.perf_counter()
    ils_stats = summarize([float(s) for s in ils_scores], t1 - t0)

    # ---- GA ----
    ga_scores: List[int] = []
    ga_best = (float("inf"), None, None)  # (score, seed, cfg)
    t2 = time.perf_counter()
    for sd in seeds:
        score, _ = run_ga(inst, ga_cfg, sd)
        ga_scores.append(score)
        if score < ga_best[0]:
            ga_best = (score, sd, ga_cfg.copy())
    t3 = time.perf_counter()
    ga_stats = summarize([float(s) for s in ga_scores], t3 - t2)

    return {
        "instance": (inst.name, inst.n_jobs, inst.n_machines),
        "repeats": repeats,
        "seeds": seeds,
        "ils": {"cfg": ils_cfg, "scores": ils_scores, "stats": ils_stats, "best": ils_best},
        "ga": {"cfg": ga_cfg, "scores": ga_scores, "stats": ga_stats, "best": ga_best},
    }


def print_report(report: Dict[str, Any]):
    name, nj, nm = report["instance"]
    print(f"\n=== INSTANCE {name} ({nj} jobs, {nm} machines) ===")
    print(f"repeats: {report['repeats']} seeds: [{report['seeds'][0]}..{report['seeds'][-1]}]")

    # ILS
    ils = report["ils"]
    ils_stats = ils["stats"]
    ils_best_score, ils_best_seed, ils_best_params = ils["best"]
    print("\n--- ILS ---")
    print("cfg:", ils["cfg"])
    print("scores:", ils["scores"])
    print("stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ils_stats.items()})
    print("best:", {"score": ils_best_score, "seed": ils_best_seed, "params": asdict(ils_best_params)})

    # GA
    ga = report["ga"]
    ga_stats = ga["stats"]
    ga_best_score, ga_best_seed, ga_best_cfg = ga["best"]
    print("\n--- GA ---")
    print("cfg:", ga["cfg"])
    print("scores:", ga["scores"])
    print("stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ga_stats.items()})
    print("best:", {"score": ga_best_score, "seed": ga_best_seed, "cfg": ga_best_cfg})


# -----------------------
# Main
# -----------------------

def main():
    jobshop = open("jobshop.txt", "r", encoding="utf-8").read()
    instances = parse_all_abz(jobshop)

    # Choose ONE:
    inst = pick_instance(instances, idx=79)
    # inst = pick_instance(instances, name="instance la22")

    report = compare_on_instance(
        inst,
        ils_cfg=DEFAULT_ILS_CFG,
        ga_cfg=DEFAULT_GA_CFG,
        repeats=10,       # <- how many runs per solver
        base_seed=42,
    )

    print_report(report)


if __name__ == "__main__":
    main()
