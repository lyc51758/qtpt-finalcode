from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import stitching_toy_mdp as toy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the diagnostic stitching toy experiment.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--n-traj-a", type=int, default=20)
    parser.add_argument("--n-traj-b", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--n-eval-envs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        data = toy.collect_dataset(
            n_envs=args.n_envs,
            n_traj_A_per_env=args.n_traj_a,
            n_traj_B_per_env=args.n_traj_b,
            seed=seed,
        )

        for method in ("QTPT", "QTPT-MC", "SPT"):
            print(f"\n=== seed={seed} method={method} ===", flush=True)
            model = toy.train(method, data, n_steps=args.n_steps, lr=args.lr, seed=seed)
            score = toy.evaluate(model, n_eval_envs=args.n_eval_envs, seed=1000 + seed)
            rows.append({"seed": seed, "method": method, "success_rate": score})
            print(f"RESULT seed={seed} method={method} success_rate={score:.4f}", flush=True)

    csv_path = args.out_dir / "stitching_multiseed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in ("QTPT", "QTPT-MC", "SPT"):
        values = np.array([r["success_rate"] for r in rows if r["method"] == method], dtype=float)
        summary[method] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
            "values": values.tolist(),
        }

    json_path = args.out_dir / "stitching_multiseed_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for method, stats in summary.items():
        print(
            f"{method}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"n={stats['n']}, values={stats['values']}"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
