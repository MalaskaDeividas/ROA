from pathlib import Path
import matplotlib.pyplot as plt

def _pad_to(hist, L):
    if not hist:
        return [0] * L
    return hist + [hist[-1]] * (L - len(hist))

def plot_convergence_mean(report, out_dir="logs", tag="conv"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    name, nj, nm = report["instance"]

    ils_h = report["ils"]["histories"]  # List[List[int]]
    ga_h  = report["ga"]["histories"]

    # Choose a common length (use max, pad shorter runs)
    L = max(max(len(h) for h in ils_h), max(len(h) for h in ga_h))

    ils_h = [_pad_to(h, L) for h in ils_h]
    ga_h  = [_pad_to(h, L) for h in ga_h]

    ils_mean = [sum(h[i] for h in ils_h) / len(ils_h) for i in range(L)]
    ga_mean  = [sum(h[i] for h in ga_h)  / len(ga_h)  for i in range(L)]

    xs = list(range(1, L + 1))

    plt.figure(figsize=(9, 4))
    plt.plot(xs, ils_mean, label="ILS mean best-so-far", marker="o", markersize=3)
    plt.plot(xs, ga_mean,  label="GA mean best-so-far",  marker="o", markersize=3)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Mean makespan (best-so-far)")
    plt.title(f"{name} ({nj} jobs, {nm} machines)")
    plt.legend()
    plt.tight_layout()

    p = out / f"convergence_{tag}.png"
    plt.savefig(p, dpi=160)
    plt.close()
    print("Saved:", p)