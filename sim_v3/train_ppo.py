"""Train a reaching policy with PPO (Stable-Baselines3).

    pip install mujoco gymnasium stable-baselines3
    python train_ppo.py --steps 1000000          # about 1M steps for a solid policy
    python train_ppo.py --eval runs/ppo_reach    # replay a saved policy, save reach_policy.gif
    python watch.py runs/ppo_reach               # second terminal: watch the policy live while it trains

A checkpoint is saved every 10% of training. With tensorboard installed, curves go to runs/
(tensorboard --logdir runs).
"""
import argparse, os, time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import arm_env  # registers ShoulderArmReach-v0

try:
    import tensorboard  # noqa: F401  (optional: only used for live training curves)
    TB = "runs"
except ImportError:
    TB = None


def save_checkpoint(model, venv, out):
    """Write model and normaliser to temp names, then swap them in, so a reader never sees half a file."""
    model.save(os.path.join(out, "model_tmp"))
    venv.save(os.path.join(out, "vecnorm_tmp.pkl"))
    os.replace(os.path.join(out, "vecnorm_tmp.pkl"), os.path.join(out, "vecnorm.pkl"))
    os.replace(os.path.join(out, "model_tmp.zip"), os.path.join(out, "model.zip"))


def evaluate(model, venv_norm=None, episodes=30, seed=1000, gif=None):
    env = gym.make("ShoulderArmReach-v0", render_mode="rgb_array" if gif else None)
    wins, finals, frames = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        hit = False
        for t in range(env.unwrapped.max_steps):
            o = venv_norm.normalize_obs(obs) if venv_norm is not None else obs
            act, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            hit |= info["success"]
            if gif and ep < 4 and t % 2 == 0:
                frames.append(env.render())
            if term or trunc:
                break
        wins += hit; finals.append(info["distance"])
    if gif and frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    env.close()
    return wins / episodes, float(np.median(finals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--out", default="runs/ppo_reach")
    ap.add_argument("--eval", default=None, help="path of a saved run to replay")
    ap.add_argument("--device", default="cpu", help="cpu is faster than a GPU for this small network")
    a = ap.parse_args()

    if a.eval:
        model = PPO.load(os.path.join(a.eval, "model"))
        norm = VecNormalize.load(os.path.join(a.eval, "vecnorm.pkl"), make_vec_env("ShoulderArmReach-v0", n_envs=1))
        norm.training = False
        rate, med = evaluate(model, norm, gif="reach_policy.gif")
        print(f"success {rate*100:.0f}%  median final distance {med*100:.1f} cm  -> reach_policy.gif")
        raise SystemExit

    os.makedirs(a.out, exist_ok=True)
    venv = make_vec_env("ShoulderArmReach-v0", n_envs=a.envs, vec_env_cls=SubprocVecEnv, seed=0)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, learning_rate=3e-4,
                gamma=0.98, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=TB, verbose=0, seed=0, device=a.device)
    t0, chunk, done = time.time(), max(50_000, a.steps // 10), 0
    while done < a.steps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, tb_log_name="ppo_reach")
        done += chunk
        venv.training = False
        rate, med = evaluate(model, venv, episodes=20)
        venv.training = True
        print(f"{done:>9,} steps  {time.time()-t0:6.0f} s   success {rate*100:5.1f}%   median final distance {med*100:5.1f} cm", flush=True)
        save_checkpoint(model, venv, a.out)          # watch.py picks this up while training continues
    print(f"saved to {a.out}")
