"""Turn the CadQuery arm into a MuJoCo model (MJCF) for simulation and training.

Writes  sim/meshes/*.stl  (visual meshes, each in its own joint frame, mm)
        sim/arm.xml       (MJCF: bodies, joints, actuators, collision shapes, body proxy)

Conventions
- Every body frame sits on the joint it rotates about; its x axis points along the link
  and z is the pin axis, exactly like the CAD scripts.
- World: x forward, y left, z up, origin on top of the LEFT shoulder (your body is toward -y).
  The foot sits on a 1.6 cm pad; the shoulder pitch joint is 9.5 cm above the foot.
- Yaw joint (vertical axis through the shoulder joint): positive = arm swings outward (+y). Arm straight = pointing up; positive shoulder
  angle swings it forward.
- Mass and inertia come from the CAD solids at an effective printed density, plus point
  masses for bearings/pins and placeholder motors. Change the numbers below once the
  real motors are chosen.
"""
import os, math
import numpy as np
import cadquery as cq
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

import base_hand as bh, l1_fork as l1, l2_link as l2, l3_fork as l3, yaw_base as yb

OUT = "sim_v3"
PRINT_DENSITY = 1.27e-6 * 0.65   # kg/mm^3: PETG at roughly 65 % effective fill
HW_PER_JOINT = 0.017             # kg: two 688 bearings + 8 mm shoulder bolt + nut
MOTOR_MASS = 0.15                # kg per joint motor (placeholder)
JAW_SERVO_MASS = 0.06            # kg (placeholder)
JOINT_TORQUE = dict(yaw=3.0, shoulder=12.0, elbow=7.0, wrist=3.0, hand=2.0, jaw=1.0)  # N*m actuator limits (placeholders)
LIMITS = dict(yaw=100, shoulder=125, elbow=125, wrist=90, hand=95)                     # deg, from the CAD collision checks
JAW_RANGE = (0, 75)


# ---------------------------------------------------------------- mass properties
def solid_props(wp):
    """mass (kg), com (mm), inertia about com (kg*mm^2) of a CadQuery solid at PRINT_DENSITY."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(wp.val().wrapped, props)
    vol = props.Mass()
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()
    I = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return vol * PRINT_DENSITY, np.array([c.X(), c.Y(), c.Z()]), I * PRINT_DENSITY


def combine(items):
    """items: list of (mass, com_mm, I_com_kgmm2). Returns mass, com (m), full inertia (kg m^2)."""
    M = sum(i[0] for i in items)
    com = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, com / 1000.0, I / 1e6


def point(m, xy):
    return (m, np.array([xy[0], xy[1], 0.0]), np.zeros((3, 3)))


def inertial_xml(items):
    M, com, I = combine(items)
    full = f"{I[0,0]:.6e} {I[1,1]:.6e} {I[2,2]:.6e} {I[0,1]:.6e} {I[0,2]:.6e} {I[1,2]:.6e}"
    return f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{M:.4f}" fullinertia="{full}"/>', M


# ---------------------------------------------------------------- body solids in joint frames (mm)
L1x, L2x, L3x = l1.L1_CENTER, l2.L2_CENTER, l3.L3_CENTER
bodies = {
    "housing": [("yaw_housing", yb.housing)],
    "yoke":  [("shoulder_yoke", yb.yoke)],
    "L1":    [("L1_outer", l1.outer.translate((L1x / 2, 0, 0))), ("L1_inner", l1.inner.translate((L1x / 2, 0, 0)))],
    "L2":    [("L2_link", l2.link.translate((L2x / 2, 0, 0)))],
    "L3":    [("L3_outer", l3.outer.translate((L3x / 2, 0, 0))), ("L3_inner", l3.inner.translate((L3x / 2, 0, 0)))],
    "hand":  [("hand_body", bh.hand)],
    "jaw":   [("hand_jaw", bh.jaw.translate((-bh.PIVOT[0], -bh.PIVOT[1], 0)))],
}
extras = {   # point masses (kg, xy in mm, body frame)
    "housing": [],
    "yoke": [(0.08, __import__("numpy").array([-79.0, 0.0, 0.0]), __import__("numpy").zeros((3, 3)))],   # 6001 inner races + M12 axle
    "L1":   [point(HW_PER_JOINT, (0, 0)), point(HW_PER_JOINT, (L1x, 0)),
             point(MOTOR_MASS, (0, 0)), point(MOTOR_MASS, (L1x, 0))],      # shoulder + elbow motors ride on the L1 fork
    "L2":   [],
    "L3":   [point(HW_PER_JOINT, (0, 0)), point(HW_PER_JOINT, (L3x, 0)),
             point(MOTOR_MASS, (0, 0)), point(MOTOR_MASS, (L3x, 0))],      # wrist + hand motors ride on the L3 fork
    "hand": [point(JAW_SERVO_MASS, bh.PIVOT)],
    "jaw":  [],
}

os.makedirs(f"{OUT}/meshes", exist_ok=True)
inertials, masses = {}, {}
for name, parts in bodies.items():
    items = []
    for mesh_name, solid in parts:
        cq.exporters.export(solid, f"{OUT}/meshes/{mesh_name}.stl", tolerance=0.05, angularTolerance=0.1)
        items.append(solid_props(solid))
    items += extras[name]
    inertials[name], masses[name] = inertial_xml(items)

# ---------------------------------------------------------------- MJCF
def mm(*v): return " ".join(f"{x/1000:.4f}" for x in v)


def visuals(name):
    return "\n".join(f'        <geom class="visual" mesh="{m}" material="{"tongue" if name in ("L2", "jaw") else ("body" if name in ("housing","yoke","hand") else "fork")}"/>'
                     for m, _ in bodies[name])


# Base frame -> world: base x = world up, base y = world forward, base z = world y (pin axis left/right)
BASE_QUAT = "0.5 -0.5 -0.5 -0.5"
xml = f"""<mujoco model="wearable_arm_v3">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <global offwidth="1280" offheight="720" azimuth="-125" elevation="-12"/>
  </visual>

  <default>
    <joint damping="0.3" armature="0.01"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.01 0.001" density="0"/>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>
    <default class="col"><geom group="3" rgba="0.9 0.4 0.2 0.25"/></default>
    <default class="pad"><geom group="3" friction="1.2 0.02 0.002" rgba="0.1 0.1 0.1 1"/></default>
    <default class="body_proxy"><geom contype="2" conaffinity="1" rgba="0.85 0.75 0.65 0.35" group="1"/></default>
  </default>

  <asset>
    <material name="fork" rgba="0.80 0.84 0.82 1"/>
    <material name="tongue" rgba="0.18 0.49 0.53 1"/>
    <material name="body" rgba="0.29 0.33 0.32 1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.95" rgb2="0.78 0.82 0.80" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1="0.92 0.93 0.92" rgb2="0.86 0.88 0.87"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
""" + "\n".join(f'    <mesh name="{m}" file="meshes/{m}.stl" scale="0.001 0.001 0.001"/>' for parts in bodies.values() for m, _ in parts) + f"""
  </asset>

  <worldbody>
    <light pos="0.4 -0.6 1.2" dir="-0.3 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 -1.4" size="2 2 0.01" material="grid" contype="0" conaffinity="0"/>

    <!-- Your body, as simple obstacles. Origin = top of the left shoulder; your body is toward -y. -->
    <geom name="torso"    class="body_proxy" type="box" pos="0 -0.17 -0.30" size="0.11 0.20 0.30"/>
    <geom name="head"     class="body_proxy" type="ellipsoid" pos="0 -0.17 0.20" size="0.085 0.075 0.11"/>
    <geom name="neck"     class="body_proxy" type="capsule" fromto="0 -0.17 0.0 0 -0.17 0.10" size="0.055"/>
    <geom name="backpack" class="body_proxy" type="box" pos="-0.18 -0.17 -0.27" size="0.07 0.18 0.24"/>
    <site name="chest_plane" pos="0.11 0 -0.15" size="0.004" rgba="0.7 0.3 0.1 1"/>

    <body name="housing" pos="0 0 {0.016 + yb.JOINT_ABOVE_FOOT/1000:.4f}" quat="{BASE_QUAT}">
      {inertials['housing']}
{visuals('housing')}
      <geom class="col" type="cylinder" fromto="{mm(-95,0,0,-63,0,0)}" size="0.020"/>
      <geom class="col" type="box" pos="{mm(-91,0,0)}" size="{mm(4,35,30)}"/>

      <body name="yoke">
        {inertials['yoke']}
        <joint name="yaw" axis="1 0 0" range="-{LIMITS['yaw']} {LIMITS['yaw']}"/>
{visuals('yoke')}
        <geom class="col" type="cylinder" fromto="{mm(-62,0,0,-56,0,0)}" size="0.025"/>
        <geom class="col" type="box" pos="{mm(-48,0,0)}" size="{mm(8,20,16)}"/>

      <body name="L1">
        {inertials['L1']}
        <joint name="shoulder" axis="0 0 1" range="-{LIMITS['shoulder']} {LIMITS['shoulder']}"/>
{visuals('L1')}
        <geom class="col" type="box" pos="{mm(L1x/2,0,0)}" size="{mm(L1x/2+20,20,9.5)}"/>

        <body name="L2" pos="{mm(L1x,0,0)}">
          {inertials['L2']}
          <joint name="elbow" axis="0 0 1" range="-{LIMITS['elbow']} {LIMITS['elbow']}"/>
{visuals('L2')}
          <geom class="col" type="box" pos="{mm(L2x/2,0,0)}" size="{mm(L2x/2+20,20,3)}"/>

          <body name="L3" pos="{mm(L2x,0,0)}">
            {inertials['L3']}
            <joint name="wrist" axis="0 0 1" range="-{LIMITS['wrist']} {LIMITS['wrist']}"/>
{visuals('L3')}
            <geom class="col" type="box" pos="{mm(L3x/2,0,0)}" size="{mm(L3x/2+20,20,9.5)}"/>

            <body name="hand" pos="{mm(L3x,0,0)}">
              {inertials['hand']}
              <joint name="hand" axis="0 0 1" range="-{LIMITS['hand']} {LIMITS['hand']}"/>
{visuals('hand')}
              <geom class="col" type="box" pos="{mm(5,0,0)}"     size="{mm(25,20,3)}"/>
              <geom class="col" type="box" pos="{mm(34,-7,0)}"   size="{mm(4,13,16)}"/>
              <geom class="col" type="box" pos="{mm(46,-7,13)}"  size="{mm(16,13,3)}"/>
              <geom class="col" type="box" pos="{mm(46,-7,-13)}" size="{mm(16,13,3)}"/>
              <geom class="col" type="box" pos="{mm(69,-16,0)}"  size="{mm(31,4,16)}"/>
              <geom name="pad_lower" class="pad" type="box" pos="{mm(75,-12,0)}" size="{mm(20,1,10)}"/>
              <site name="grip" pos="{mm(75,-10.5,0)}" size="0.005" rgba="0.9 0.2 0.2 1"/>

              <body name="jaw" pos="{mm(*bh.PIVOT,0)}">
                {inertials['jaw']}
                <joint name="jaw" axis="0 0 1" range="{JAW_RANGE[0]} {JAW_RANGE[1]}"/>
{visuals('jaw')}
                <geom class="col" type="box" pos="{mm(27,-1,0)}" size="{mm(27,4,9.5)}"/>
                <geom name="pad_upper" class="pad" type="box" pos="{mm(29,-5,0)}" size="{mm(20,1,8.5)}"/>
              </body>
            </body>
          </body>
        </body>
        </body>
      </body>
    </body>

    <!-- Reach target for training; the environment moves it every episode. -->
    <body name="target" mocap="true" pos="0.25 0.05 0.05">
      <geom type="sphere" size="0.012" rgba="0.9 0.3 0.1 0.6" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="yaw"      joint="yaw"      kp="25"  kv="0.8" forcerange="-{JOINT_TORQUE['yaw']} {JOINT_TORQUE['yaw']}"             ctrlrange="{-math.radians(LIMITS['yaw']):.4f} {math.radians(LIMITS['yaw']):.4f}"/>
    <position name="shoulder" joint="shoulder" kp="60"  kv="2"   forcerange="-{JOINT_TORQUE['shoulder']} {JOINT_TORQUE['shoulder']}" ctrlrange="{-math.radians(LIMITS['shoulder']):.4f} {math.radians(LIMITS['shoulder']):.4f}"/>
    <position name="elbow"    joint="elbow"    kp="40"  kv="1.2" forcerange="-{JOINT_TORQUE['elbow']} {JOINT_TORQUE['elbow']}"       ctrlrange="{-math.radians(LIMITS['elbow']):.4f} {math.radians(LIMITS['elbow']):.4f}"/>
    <position name="wrist"    joint="wrist"    kp="15"  kv="0.4" forcerange="-{JOINT_TORQUE['wrist']} {JOINT_TORQUE['wrist']}"       ctrlrange="{-math.radians(LIMITS['wrist']):.4f} {math.radians(LIMITS['wrist']):.4f}"/>
    <position name="hand"     joint="hand"     kp="10"  kv="0.3" forcerange="-{JOINT_TORQUE['hand']} {JOINT_TORQUE['hand']}"         ctrlrange="{-math.radians(LIMITS['hand']):.4f} {math.radians(LIMITS['hand']):.4f}"/>
    <position name="jaw"      joint="jaw"      kp="5"   kv="0.1" forcerange="-{JOINT_TORQUE['jaw']} {JOINT_TORQUE['jaw']}"           ctrlrange="{math.radians(JAW_RANGE[0]):.4f} {math.radians(JAW_RANGE[1]):.4f}"/>
  </actuator>

  <sensor>
    <jointpos joint="yaw"/><jointpos joint="shoulder"/><jointpos joint="elbow"/><jointpos joint="wrist"/><jointpos joint="hand"/><jointpos joint="jaw"/>
    <actuatorfrc actuator="yaw"/><actuatorfrc actuator="shoulder"/><actuatorfrc actuator="elbow"/><actuatorfrc actuator="wrist"/><actuatorfrc actuator="hand"/>
    <framepos objtype="site" objname="grip"/>
  </sensor>
</mujoco>
"""
with open(f"{OUT}/arm.xml", "w") as f:
    f.write(xml)

if __name__ == "__main__":
    for k, v in masses.items():
        print(f"{k:5s} mass {v*1000:6.1f} g")
    print(f"moving arm total (yoke onward): {sum(v for k, v in masses.items() if k != 'housing')*1000:.0f} g")
