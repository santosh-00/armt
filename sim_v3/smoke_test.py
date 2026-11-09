"""Check the environment works and that its targets are reachable.

1. Gymnasium's API checker.
2. A random policy (baseline a trained policy must beat).
3. A hand-written Jacobian controller (shows every target is reachable, so the task is solvable).
"""
import numpy as np, mujoco, gymnasium as gym
from gymnasium.utils.env_checker import check_env
import arm_env  # registers ShoulderArmReach-v0

env = gym.make("ShoulderArmReach-v0")
check_env(env.unwrapped, skip_render_check=True)
print("gymnasium check_env: passed")


def run(policy, episodes=30, seed=0):
    succ, final = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        hit, touch = False, False
        for _ in range(env.unwrapped.max_steps):
            obs, r, term, trunc, info = env.step(policy(env.unwrapped))
            hit |= info["success"]; touch |= info["touching_body"]
            if term or trunc:
                break
        succ += hit; final.append(info["distance"])
    return succ / episodes, float(np.median(final))


def random_policy(e):
    return e.action_space.sample()


def jacobian_policy(e, gain=0.6):
    m, d = e.model, e.data
    jac = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jac, None, e.grip)
    J = jac[:, :e.n_arm]                                 # x, y, z rows; yaw..hand columns
    err = d.mocap_pos[e.target_mocap] - d.site_xpos[e.grip]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)   # damped least squares
    return np.clip(gain * dq / e.step_rad, -1, 1)


for name, pol in [("random policy", random_policy), ("Jacobian controller", jacobian_policy)]:
    rate, med = run(pol)
    print(f"{name:20s} success {rate*100:5.1f}%   median final distance {med*100:5.1f} cm")
