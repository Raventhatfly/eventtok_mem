"""Can the RoboMME simulator run here, and what does it hand a policy?

    python -m eventtok.scripts.sim_smoke --task SwingXtimes

Gating check before building the rollout adapter: confirm the env constructs on a GPU
node, print the observation keys and shapes a policy will receive, and confirm the
action space matches the 8-dimensional end-effector-pose-plus-gripper actions the
dataset and the trained policy use.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

BENCH_SRC = "/n/home04/wfy/repos/robomme_policy_learning/third_party/robomme_benchmark/src"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SwingXtimes")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--action-space", default="ee_pose")
    ap.add_argument("--bench-src", default=BENCH_SRC)
    args = ap.parse_args()

    if args.bench_src not in sys.path:
        sys.path.insert(0, args.bench_src)
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    print("tasks:", BenchmarkEnvBuilder.get_task_list(), flush=True)
    builder = BenchmarkEnvBuilder(
        env_id=args.task, dataset="test", action_space=args.action_space, max_steps=1300
    )
    print("episodes:", builder.get_episode_num(), flush=True)

    env = builder.make_env_for_episode(args.episode)
    obs, info = env.reset()
    print("task_goal:", info["task_goal"][0], flush=True)
    print("action_space:", env.action_space, flush=True)
    for k, v in obs.items():
        try:
            a = np.asarray(v)
            print(f"  obs[{k}] {a.shape} {a.dtype}", flush=True)
        except Exception:
            print(f"  obs[{k}] {type(v).__name__}", flush=True)

    front = np.asarray(obs["front_rgb_list"][-1])
    wrist = np.asarray(obs["wrist_rgb_list"][-1])
    print(f"front {front.shape} {front.dtype}   wrist {wrist.shape}", flush=True)

    dim = env.action_space.shape[-1] if env.action_space.shape else 8
    print(f"action dim {dim}", flush=True)
    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(np.zeros(dim, dtype=np.float32))
        status = info.get("status") if isinstance(info, dict) else None
        print(
            f"  step {i}: reward={float(np.asarray(r).mean()):.4f} "
            f"term={bool(np.asarray(term).any())} trunc={bool(np.asarray(trunc).any())} "
            f"status={status}",
            flush=True,
        )
        if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
            break
    env.close()
    print("SIM OK", flush=True)


if __name__ == "__main__":
    main()
