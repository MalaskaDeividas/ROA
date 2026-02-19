from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

from parser.abz_parser import parse_all_abz
from trajectory.ils import ILSParams, iterated_local_search
from population.gs import Population
from datetime import datetime
from pathlib import Path
import itertools
from dataclasses import dataclass


def solve_ils(instance, seed, cfg):
    params = ILSParams(
        iterations=cfg["iterations"],
        local_steps=cfg["local_steps"],
        swaps=cfg["swaps"],
        seed=seed,
        accept_worse_prob=cfg["accept_worse_prob"],
        verbose_every=0,
        init=cfg["init"],
    )
    res = iterated_local_search(instance, params)
    return res.best_makespan

def solve_ga(instance, seed, cfg):
    
    random.seed(seed)

    pop = Population(
        problem_instance=instance,
        lambda_param=cfg["lambda"], 
        mu_param=cfg["mu"],
        max_iterations_without_improvement=cfg["max_no_improve"],
        new_genomes_per_gen=cfg["new_genomes_per_gen"],
    )
    best = pop.simulation()
    return best


SOLVERS = {
    "ils": solve_ils,
    "ga": solve_ga,
}


# ---------------------------
#           Config 
# ---------------------------

def sample_ils_cfg(rng):
    return {
        "iterations": rng.choice([87]),
        "local_steps": rng.choice([200]),
        "swaps": rng.choice([2]),
        "accept_worse_prob": rng.choice([0.15]),
        "init": rng.choice(["order"]),
    }


def sample_ga_cfg(rng):
    lambda_param = rng.randint(50, 200)
    mu_param = rng.randint(lambda_param * 2, 1000)
    return {
        "lambda": lambda_param,
        "mu": mu_param,
        "new_genomes_per_gen": rng.choice([150]),
        "max_no_improve": rng.choice([5,10,15,20]),
    }


CFG_SAMPLERS = {
    "ils": sample_ils_cfg,
    "ga": sample_ga_cfg,
}


def evaluate_config(instances, solver_name, cfg, seeds, instance_indices):
    solve_fn = SOLVERS[solver_name]

    scores = []
    t0 = time.perf_counter()

    for idx in instance_indices:
        inst = instances[idx]
        for sd in seeds:
            score = solve_fn(inst, sd, cfg)
            scores.append(score)

    t1 = time.perf_counter()

    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "runtime_sec": t1 - t0,
        "n_runs": len(scores),
    }


def random_search(instances, solver_name, n_trials: int = 30, log=print):
    rng = random.Random(1234)
    sample_cfg = CFG_SAMPLERS[solver_name]

    # Tune on a small subset first (fast)
    instance_indices = [0, 1, 5, 10, 20]
    # Fixed seeds for fairness across configs
    seeds = [1, 2, 3]

    best = None

    for t in range(n_trials):
        cfg = sample_cfg(rng)
        stats = evaluate_config(instances, solver_name, cfg, seeds, instance_indices)

        print(
            f"[{solver_name.upper()} {t+1:02d}/{n_trials}] "
            f"mean={stats['mean']:.2f} median={stats['median']:.2f} "
            f"time={stats['runtime_sec']:.2f}s cfg={cfg}"
        )

        key = (stats["mean"], stats["median"], stats["runtime_sec"])
        if best is None or key < best["key"]:
            best = {"key": key, "cfg": cfg, "stats": stats}

    return best


def print_best_usage(solver_name, best, log):
    cfg = best["cfg"]
    log("\nBEST CONFIG FOUND")
    log(f"cfg: {cfg}")
    log(f"stats: {best['stats']}")

    log("\nCOPY/PASTE USAGE:")
    if solver_name == "ils":
        log(
            "params = ILSParams("
            f"iterations={cfg['iterations']}, "
            f"local_steps={cfg['local_steps']}, "
            f"swaps={cfg['swaps']}, "
            "seed=123, "
            f"accept_worse_prob={cfg['accept_worse_prob']}, "
            "verbose_every=0, "
            f"init='{cfg['init']}'"
            ")"
        )
    elif solver_name == "ga":
        log(
            "population = Population("
            "problem_instance=inst, "
            f"lambda={cfg['lambda']}, "
            f"mu={cfg['mu']}, "
            f"max_iterations_without_improvement={cfg['max_no_improve']}, "
            f"new_genomes_per_gen={cfg['new_genomes_per_gen']}"
            ")"
        )


def load_instances() -> list:
    root = Path(__file__).resolve().parents[1]
    jobshop_path = root / "jobshop.txt"
    text = jobshop_path.read_text(encoding="utf-8")
    return parse_all_abz(text)


def next_log_path(root: Path) -> Path:
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    k = 0
    while True:
        p = logs_dir / f"tune_{k}.txt"
        if not p.exists():
            return p
        k += 1

def run_tuner():
    instances = load_instances()

    root = Path(__file__).resolve().parents[1]
    log_path = next_log_path(root)

    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg)
            f.write(str(msg) + "\n")
            f.flush()

        log(f"Log file: {log_path}")

        for solver_name in ["ga", "ils"]:
            best = random_search(instances, solver_name, n_trials=30, log=log)
            print_best_usage(solver_name, best, log=log)

    print(f"\nSaved tuning log to: {log_path}")


