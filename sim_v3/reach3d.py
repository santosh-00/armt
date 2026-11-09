"""Sample the v3 arm's workspace with the real MuJoCo model (joint limits + body collisions)."""
import numpy as np, mujoco
m = mujoco.MjModel.from_xml_path("arm.xml"); d = mujoco.MjData(m)
rng = np.random.default_rng(0); g = m.site("grip").id
lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
N = 200_000; P = []
for _ in range(N):
    d.qpos[:5] = rng.uniform(lo, hi); d.qpos[5] = 0.5
    mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
    if d.ncon == 0:
        P.append(d.site_xpos[g].copy())
P = np.array(P); np.save("reach3d.npy", P)
CH = 0.11                                   # chest plane (x), shoulder top z = 0, body toward -y
front = P[P[:, 0] > CH]
vox = np.unique(np.floor(front / 0.02).astype(int), axis=0)
print(f"valid poses {len(P)/N*100:.0f}%   reachable volume in front of chest ~{len(vox)*8/1000:.1f} litres (2 cm voxels)")
print("height (cm vs shoulder top) | furthest forward past chest | sideways span in front (y, cm; - = toward your body)")
for z in (0.30, 0.20, 0.10, 0.0, -0.10, -0.20, -0.30):
    b = front[np.abs(front[:, 2] - z) < 0.01]
    if len(b):
        print(f"  {z*100:+5.0f}                     | {(b[:,0].max()-CH)*100:5.1f} cm                    | {b[:,1].min()*100:+6.1f} to {b[:,1].max()*100:+6.1f}")
    else:
        print(f"  {z*100:+5.0f}                     | not reachable")
