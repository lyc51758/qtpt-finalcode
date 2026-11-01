import pickle
import numpy as np
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qtpt_core", "data", "darkroom")

for split in ["train", "test"]:
    f0 = os.path.join(data_dir, f"trajs_darkroom_envs100000_hists1_samples1_H100_d10_mix0.0_{split}.pkl")
    f1 = os.path.join(data_dir, f"trajs_darkroom_envs100000_hists1_samples1_H100_d10_mix1.0_{split}.pkl")
    
    with open(f0, "rb") as fh:
        data0 = pickle.load(fh)
    with open(f1, "rb") as fh:
        data1 = pickle.load(fh)
    
    print(f"{split}: random has {len(data0)} trajs, expert has {len(data1)} trajs")
    
    # Take 50% from each
    n = len(data0)
    half = n // 2
    np.random.seed(42)
    idx0 = np.random.choice(n, half, replace=False)
    idx1 = np.random.choice(len(data1), n - half, replace=False)
    
    mixed = [data0[i] for i in idx0] + [data1[i] for i in idx1]
    np.random.shuffle(mixed)
    
    out_path = os.path.join(data_dir, f"trajs_darkroom_envs100000_hists1_samples1_H100_d10_mix0.5_{split}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(mixed, fh)
    print(f"Saved {len(mixed)} trajs to {out_path}")

# Also create eval file by mixing eval files
f0_eval = os.path.join(data_dir, "trajs_darkroom_envs100_H100_d10_mix0.0_eval.pkl")
f1_eval = os.path.join(data_dir, "trajs_darkroom_envs100_H100_d10_mix1.0_eval.pkl")
if os.path.exists(f0_eval) and os.path.exists(f1_eval):
    with open(f0_eval, "rb") as fh:
        eval0 = pickle.load(fh)
    with open(f1_eval, "rb") as fh:
        eval1 = pickle.load(fh)
    n_eval = len(eval0)
    half_eval = n_eval // 2
    mixed_eval = eval0[:half_eval] + eval1[:n_eval - half_eval]
    out_eval = os.path.join(data_dir, "trajs_darkroom_envs100_H100_d10_mix0.5_eval.pkl")
    with open(out_eval, "wb") as fh:
        pickle.dump(mixed_eval, fh)
    print(f"Saved {len(mixed_eval)} eval trajs to {out_eval}")

print("Done!")
