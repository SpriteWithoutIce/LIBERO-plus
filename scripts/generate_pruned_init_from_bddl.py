#!/usr/bin/env python3
"""Generate dedicated .pruned_init files from task BDDL files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_libero_config():
    cfg_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(REPO_ROOT / ".libero")))
    cfg_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_root)
    cfg_file = cfg_root / "config.yaml"
    if cfg_file.exists():
        return

    libero_root = REPO_ROOT / "libero" / "libero"
    default_cfg = {
        "benchmark_root": str(libero_root),
        "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"),
        "datasets": str(libero_root.parent / "datasets"),
        "assets": str(libero_root / "assets"),
    }
    with open(cfg_file, "w") as f:
        yaml.dump(default_cfg, f)


_ensure_libero_config()
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def collect_init_states(bddl_file: Path, num_states: int, base_seed: int) -> np.ndarray:
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=64,
        camera_widths=64,
    )
    states = []
    try:
        for i in range(num_states):
            env.seed(base_seed + i)
            env.reset()
            states.append(env.get_sim_state().copy())
    finally:
        env.close()
    return np.stack(states, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", required=True, help="task name without .bddl")
    parser.add_argument("--num-states", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--bddl-root", default="libero/libero/bddl_files")
    parser.add_argument("--init-root", default="libero/libero/init_files")
    args = parser.parse_args()

    bddl_file = Path(args.bddl_root) / args.suite / f"{args.task}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Missing bddl file: {bddl_file}")

    init_dir = Path(args.init_root) / args.suite
    init_dir.mkdir(parents=True, exist_ok=True)
    init_file = init_dir / f"{args.task}.pruned_init"

    states = collect_init_states(bddl_file, num_states=args.num_states, base_seed=args.base_seed)
    torch.save(states, init_file)

    print(f"Saved {states.shape[0]} states to {init_file}")
    print("shape:", states.shape)


if __name__ == "__main__":
    main()
