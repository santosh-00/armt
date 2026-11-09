# Wearable arm v3 (yaw base): simulation model and training

The CAD arm as a MuJoCo model (MJCF), mounted on top of your LEFT shoulder on a yaw turntable,
so it reaches points in 3D in front of you. A simple body sits around it, and a Gymnasium
environment trains it to reach targets anywhere in that space.

## Files

| File | What it is |
|---|---|
| `arm.xml` | The model: housing, yoke, L1, L2, L3, hand, jaw; 5 arm joints (yaw, shoulder, elbow, wrist, hand) + jaw; position motors; collision shapes; your body as obstacles |
| `meshes/*.stl` | Visual meshes exported from the CAD, one per part, each in its joint's frame |
| `check_model.py` | Holding torques, tracking, and a test that the arm stops against your chest |
| `arm_env.py` | `ShoulderArmReach-v0`: reach a 3D target in front of you (5 actions: yaw..hand) |
| `smoke_test.py` | Gymnasium API check, random baseline, and a Jacobian controller that proves targets are reachable |
| `train_ppo.py` | PPO training with Stable-Baselines3, plus `--eval` to replay a policy as a GIF |
| `build_sim_v3.py`, `yaw_base.py` | Regenerate `arm.xml` and the meshes from the CAD scripts (needs the CAD folder) |
| `watch.py` | Plays the newest checkpoint live in the MuJoCo viewer while training runs |
| `reach3d.py` | Samples the reachable space and prints reach per height |

## Quick start

```bash
pip install mujoco gymnasium stable-baselines3
python -m mujoco.viewer --mjcf=arm.xml        # look at it, drag the joints in the Control panel
python check_model.py
python smoke_test.py
python train_ppo.py --steps 1000000            # 8 parallel envs by default
python train_ppo.py --eval runs/ppo_reach      # writes reach_policy.gif
```

## Included trained policy

See the end of this file for the 3D policy in `runs/ppo_reach3d/`.

## How the model is put together

- **Frames.** Each body's frame sits on the joint it turns about, with x along the link and
  z along the pin, exactly as in the CAD. World: x forward, y left, z up, origin on top of
  your left shoulder (your body is toward -y). Arm straight = pointing up; a positive shoulder
  angle swings it forward.
- **Yaw.** A vertical axis through the shoulder joint, ±100°. Positive swings the arm outward
  (+y), negative swings it across in front of you toward your head. Gravity acts along this
  axis, so the yaw motor only fights inertia, not weight.
- **Mass.** Mass, centre of mass and inertia come from the CAD solids at 65 % of PETG's
  density, plus 17 g of bearings and bolts per joint and **placeholder motors of 150 g**
  at each joint (60 g for the jaw servo). Everything that turns on the yaw axis comes to 999 g.
- **Joint limits** are the ones measured on the CAD: yaw ±100°, shoulder and elbow ±125°,
  wrist ±90°, hand ±95°, jaw 0–75°.
- **Motors** are position servos with **placeholder torque limits**: yaw 3, shoulder 12, elbow 7,
  wrist 3, hand 2, jaw 1 N·m. With 1.5 kg held straight out, the model needs 7.97 N·m at
  the shoulder and 4.54 N·m at the elbow, matching the hand calculation.
- **Collisions.** Contacts use simple boxes (shown in group 3 in the viewer), not the meshes.
  MuJoCo collides meshes by their convex hull, which would fill the gap between the jaws
  and the fork gaps. Pads have higher friction (1.2) for grasping.
- **Your body** is a torso box, neck, head and backpack. With yaw the arm can swing in toward
  your head; the head and neck stop it in simulation, and the policy is penalised for touching them. They are fixed obstacles; touching
  them costs reward in the environment.

## When the real motors are picked

Edit the top of `build_sim.py` (`MOTOR_MASS`, `JOINT_TORQUE`) and run it again, or edit
`arm.xml` directly: change `forcerange` on each `<position>` actuator and the motor masses
inside each body's `<inertial>`. Also set `armature` (reflected rotor inertia) and `damping`
from the motor datasheet so the sim behaves like the hardware.

## Using it in other simulators

- **MuJoCo MJX / MuJoCo Playground (JAX, GPU):** loads `arm.xml` directly for thousands of
  parallel envs. Swap the box contacts for primitives MJX supports well if needed.
- **NVIDIA Newton / Isaac Lab:** both import MJCF. Check their importer docs for how they map
  `position` actuators and `mocap` bodies; the target body may need replacing with a marker.

## Ideas for the next environments

1. **Body motion.** Turn the `base` into a mocap body and replay recorded IMU motion
   (walking, turning) so the policy learns to hold a target steady while you move.
2. **Point a flashlight.** Reward the angle between the hand's x axis and a direction, not a position.
3. **Grasp and hand over.** Add a free box between the pads, reward lifting it to your hand position.
4. **Domain randomisation.** Randomise link masses, motor strength and friction each episode
   so the policy survives the move to the real arm.

## Included 3D policy

`runs/ppo_reach3d/` holds a PPO policy trained for 1M steps (about 15 minutes on 2 CPU cores).
On 50 unseen targets spread from 27 cm toward your head to 33 cm outward, it reached 98%,
finishing a median 0.5 cm away; one episode brushed your body. See `reach_policy_3d.gif`.
Replay it: `python train_ppo.py --eval runs/ppo_reach3d` or watch live: `python watch.py runs/ppo_reach3d`.
