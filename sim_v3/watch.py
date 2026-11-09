"""Watch the policy live in the MuJoCo viewer while it trains.

    python watch.py runs/ppo_left

Plays one episode after another at real-time speed with a new target each time, and
reloads the newest checkpoint whenever train_ppo.py saves one (every 10% of training),
so you see the arm get better as training goes on. Close the window to stop.
"""
import os, sys, time
import numpy as np
import mujoco, mujoco.viewer
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import arm_env  # registers ShoulderArmReach-v0

run = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_reach"
model_path, norm_path = os.path.join(run, "model.zip"), os.path.join(run, "vecnorm.pkl")

env = gym.make("ShoulderArmReach-v0")
e = env.unwrapped
dummy = make_vec_env("ShoulderArmReach-v0", n_envs=1)


def load_latest(last_mtime):
    """Load the checkpoint if it is newer than the one we have. Returns (model, norm, mtime) or None."""
    if not (os.path.exists(model_path) and os.path.exists(norm_path)):
        return None
    mtime = os.path.getmtime(model_path)
    if mtime == last_mtime:
        return None
    try:
        model = PPO.load(model_path, device="cpu")
        norm = VecNormalize.load(norm_path, dummy)
        norm.training = False
        return model, norm, mtime
    except Exception:          # file being written right now: try again next episode
        return None


print(f"Waiting for the first checkpoint in {run} ...")
model = norm = None
mtime = None
while model is None:
    got = load_latest(mtime)
    if got:
        model, norm, mtime = got
    else:
        time.sleep(2)
version = 1
print("Loaded checkpoint 1. Close the viewer window to stop.")

with mujoco.viewer.launch_passive(e.model, e.data) as viewer:
    viewer.cam.lookat[:] = [0.1, -0.1, 0.1]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, -55, -12
    episode = 0
    while viewer.is_running():
        got = load_latest(mtime)
        if got:
            model, norm, mtime = got
            version += 1
            print(f"Loaded checkpoint {version}")
        obs, _ = env.reset(seed=10_000 + episode)
        hits = 0
        for _ in range(e.max_steps):
            t0 = time.time()
            act, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            hits += info["success"]
            viewer.sync()
            if not viewer.is_running():
                break
            time.sleep(max(0.0, e.frame_skip * e.model.opt.timestep - (time.time() - t0)))  # real time
            if term or trunc:
                break
        print(f"checkpoint {version}  episode {episode:3d}  final distance {info['distance']*100:5.1f} cm"
              f"  {'reached' if hits else 'missed'}{'  (touched you)' if info['touching_body'] else ''}")
        episode += 1
