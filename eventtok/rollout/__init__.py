"""Closed-loop evaluation in the RoboMME simulator.

Everything else in this project measures action prediction on recorded episodes. That
is open loop: the policy is scored against a demonstrator's actions from the
demonstrator's own state distribution, and never has to live with its own mistakes.
Success rate is the number the project's claims actually rest on, and this package is
what produces it.

The simulator is ManiSkill/SAPIEN, registered by
``robomme_policy_learning/third_party/robomme_benchmark/src`` -- not robosuite or
mujoco, which is what an earlier check looked for before wrongly concluding no
simulator was available here.
"""
