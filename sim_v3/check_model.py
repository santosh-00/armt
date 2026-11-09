"""Sanity checks for arm.xml (v3, with yaw): holding torques, tracking, contacts. Run: python check_model.py"""
import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("arm.xml")
d = mujoco.MjData(m)
grip = m.site("grip").id
J = ["yaw", "shoulder", "elbow", "wrist", "hand", "jaw"]


def holding_torque(q_deg, payload_kg):
    d.qpos[:] = np.radians(q_deg); d.qvel[:] = 0; d.qacc[:] = 0
    mujoco.mj_inverse(m, d)                       # torque to hold the arm itself
    tau = d.qfrc_inverse.copy()
    jacp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, None, grip)
    tau += -jacp.T @ np.array([0, 0, -payload_kg * 9.81])   # plus the payload at the grip point
    return tau


def contacts():
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2)}
    return sorted(n for n in names if n) or ["nothing"]


def drive(ctrl_deg, seconds=2.0):
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = np.radians(ctrl_deg)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


print("Holding torque, arm straight out forward (worst case), N*m")
for kg in (0.0, 1.5):
    for yaw in (0, 60):
        t = holding_torque([yaw, 90, 0, 0, 0, 0], kg)
        print(f"  payload {kg:.1f} kg, yaw {yaw:>2}: " + "  ".join(f"{n} {t[i]:6.2f}" for i, n in enumerate(J[:5])))
print("  actuator limits:          " + "  ".join(f"{n} {m.actuator_forcerange[i,1]:6.2f}" for i, n in enumerate(J[:5])))

mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
print(f"\nShoulder joint at {d.xpos[m.body('L1').id].round(3)} m, grip at rest {d.site_xpos[grip].round(3)} m")

drive([40, 60, 50, -30, 20, 30])
err = np.degrees(d.qpos - np.radians([40, 60, 50, -30, 20, 30]))
print("\nTracking after 2 s (deg error): " + "  ".join(f"{n} {e:+.2f}" for n, e in zip(J, err)))
print("Contacts:", ", ".join(contacts()))

drive([-100, 30, 0, 0, 0, 0])
print(f"\nYawing in toward your head: touches {contacts()}; yaw stops at {np.degrees(d.qpos[0]):.1f} deg")
drive([0, 125, 125, 0, 0, 0])
print(f"Folding down into your chest: touches {contacts()}; shoulder/elbow stop at {np.degrees(d.qpos[1:3]).round(1)} deg")
