"""Gymnasium environment: the shoulder-mounted arm reaches for targets in front of you.

    import gymnasium as gym, arm_env
    env = gym.make("ShoulderArmReach-v0")

Observation (21): joint angles (6), joint velocities (6), grip position (3),
                  target position (3), target - grip (3). Metres, radians.
Action (5):       change in the target angle of yaw, shoulder, elbow, wrist, hand,
                  each in [-1, 1], scaled to at most 3 degrees per step. The jaw stays open.
Targets are anywhere in 3D in front of you, not just in one plane.
Reward:           -distance(grip, target), a small effort penalty, -1 for touching
                  your body, +1 while the grip is within 1.5 cm of the target.
Episodes:         150 steps of 0.04 s (6 s). Targets are drawn from poses the arm can
                  actually reach in front of your chest without touching you.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm.xml")
BODY_GEOMS = ("torso", "head", "neck", "backpack")


class ShoulderArmReach(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, frame_skip=20, max_steps=150, step_deg=3.0,
                 success_radius=0.015, chest_x=0.11):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.max_steps = frame_skip, max_steps
        self.step_rad = np.radians(step_deg)
        self.success_radius, self.chest_x = success_radius, chest_x
        self.render_mode, self._renderer = render_mode, None

        m = self.model
        self.grip = m.site("grip").id
        self.target_mocap = m.body_mocapid[m.body("target").id]
        self.n_arm = 5                                   # yaw, shoulder, elbow, wrist, hand
        self.arm_act = np.arange(self.n_arm)
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self.body_geoms = {m.geom(n).id for n in BODY_GEOMS}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_arm,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * m.nq + 9,), dtype=np.float64)

    # ------------------------------------------------------------ helpers
    def _touching_body(self):
        d = self.data
        return any((c.geom1 in self.body_geoms) != (c.geom2 in self.body_geoms) for c in d.contact[:d.ncon])

    def _sample_target(self):
        """Pick a random collision-free pose, keep its grip point if it is in front of the chest."""
        m, d = self.model, self.data
        lo, hi = m.jnt_range[:self.n_arm, 0], m.jnt_range[:self.n_arm, 1]
        q_save = d.qpos.copy()
        for _ in range(500):
            d.qpos[:self.n_arm] = self.np_random.uniform(lo, hi)
            mujoco.mj_forward(m, d)
            p = d.site_xpos[self.grip].copy()
            if p[0] > self.chest_x + 0.02 and d.ncon == 0:
                d.qpos[:] = q_save
                mujoco.mj_forward(m, d)
                return p
        d.qpos[:] = q_save
        mujoco.mj_forward(m, d)
        return np.array([0.25, 0.05, 0.10])

    def _obs(self):
        d = self.data
        g = d.site_xpos[self.grip]
        t = d.mocap_pos[self.target_mocap]
        return np.concatenate([d.qpos, d.qvel, g, t, t - g])

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:self.n_arm] = self.np_random.uniform(-0.3, 0.3, self.n_arm)   # start near straight up
        d.qpos[self.n_arm] = np.radians(30)                                  # jaw open
        d.ctrl[:] = d.qpos[:m.nu]
        mujoco.mj_forward(m, d)
        d.mocap_pos[self.target_mocap] = self._sample_target()
        mujoco.mj_forward(m, d)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        d.ctrl[self.arm_act] = np.clip(d.ctrl[self.arm_act] + a * self.step_rad,
                                       self.ctrl_lo[self.arm_act], self.ctrl_hi[self.arm_act])
        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self.steps += 1

        dist = float(np.linalg.norm(d.mocap_pos[self.target_mocap] - d.site_xpos[self.grip]))
        touching = self._touching_body()
        success = dist < self.success_radius
        reward = -dist - 0.01 * float(a @ a) - (1.0 if touching else 0.0) + (1.0 if success else 0.0)
        truncated = self.steps >= self.max_steps
        info = {"distance": dist, "success": success, "touching_body": touching}
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0.1, -0.1, 0.05]; self._cam.distance = 1.2
            self._cam.azimuth, self._cam.elevation = -125, -12
            self._opt = mujoco.MjvOption(); self._opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self._renderer.update_scene(self.data, self._cam, self._opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="ShoulderArmReach-v0", entry_point=ShoulderArmReach)
gym.register(id="ShoulderArmReach3D-v0", entry_point=ShoulderArmReach)
