"""Can SAPIEN create a renderer on this node? Probes what blocks the rollouts.

    python -m eventtok.scripts.probe_vulkan

The RoboMME simulator constructs and registers all 16 tasks, but resetting an
environment dies with ``vk::createInstanceUnique: ErrorIncompatibleDriver``. Rollouts
are the only measurement that can distinguish "the event log helps a policy act" from
"the event log improves action prediction", so it is worth knowing whether this is a
property of one node type or of the cluster.

Prints the GPU, the driver, and whether a SAPIEN render system initialises, so the same
script can be fired at several partitions and the results compared.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    print(f"host {os.uname().nodename}", flush=True)
    for cmd in (
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        ["bash", "-lc", "ls /usr/share/vulkan/icd.d/ 2>/dev/null | tr '\\n' ' '"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
            print(f"  {cmd[0]}: {out or '(nothing)'}", flush=True)
        except Exception as exc:
            print(f"  {cmd[0]}: {exc}", flush=True)

    # Probe the path that actually matters: constructing and resetting a RoboMME
    # environment. An earlier version of this called sapien.render.RenderSystem(0),
    # which is the wrong constructor signature, so it reported a TypeError from this
    # script as though it were a driver failure.
    sys.path.insert(0, "/n/home04/wfy/repos/robomme_policy_learning/"
                       "third_party/robomme_benchmark/src")
    for gl in ("egl", "osmesa", None):
        env = "default" if gl is None else gl
        if gl:
            os.environ["MUJOCO_GL"] = gl
            os.environ["PYOPENGL_PLATFORM"] = gl
        code = (
            "import os,sys;"
            "sys.path.insert(0,'/n/home04/wfy/repos/robomme_policy_learning/"
            "third_party/robomme_benchmark/src');"
            "from robomme.env_record_wrapper import BenchmarkEnvBuilder as B;"
            "e=B(env_id='SwingXtimes',dataset='test',action_space='ee_pose',"
            "max_steps=50).make_env_for_episode(0);"
            "o,i=e.reset();print('RESET OK', i['task_goal'][0][:40]);e.close()"
        )
        # A separate process each time: a failed Vulkan init can leave the module
        # unusable, so retrying in-process would report the first failure forever.
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=900,
                           env={**os.environ, "MUJOCO_GL": gl or "", 
                                "PYOPENGL_PLATFORM": gl or ""})
        if "RESET OK" in r.stdout:
            print(f"  MUJOCO_GL={env}: {r.stdout.strip().splitlines()[-1]}", flush=True)
            print("VULKAN OK", flush=True)
            return
        tail = [l for l in (r.stderr or "").splitlines()
                if l.strip() and "warn" not in l.lower()][-1:] or ["(no error text)"]
        print(f"  MUJOCO_GL={env}: {tail[0][:130]}", flush=True)

    print("VULKAN BLOCKED", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
