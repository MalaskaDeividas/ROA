from pathlib import Path










def main():
    root = Path(__file__).resolve().parents[1]   # ROA/
    jobshop_path = root / "jobshop.txt"

    with open(jobshop_path, "r", encoding="utf-8") as f:
        jobshop = f.read()

    instances = parse_all_abz(jobshop)

    best = random_search(instances, n_trials=30)

    print("\nBEST CONFIG FOUND")
    print("cfg:", best["cfg"])
    print("stats:", best["stats"])

