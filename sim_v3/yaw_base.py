"""Yaw turntable base: lets the whole arm swing left/right about a vertical axis.

Replaces base_mount. Two printed parts plus small printed spacers:
  yaw_housing   fixed. Foot plate (4 x M4 to your shoulder plate) + tube with two 6001
                bearings (12 x 28 x 8) spaced 24 mm apart, so the arm's tipping moment
                becomes a pair of radial loads instead of bending one small bearing.
  shoulder_yoke turns on the yaw axis. Hub + block + the same 6 mm shoulder tongue as before,
                so L1's fork drops on exactly as it did on base_mount.
  yaw_spacer    tube between the two inner races.  yaw_collar  under the bolt head.

Axle: M12 x 70 hex bolt, partly threaded (30 mm thread), cut to 55 mm. Its plain shank runs
through both bearings; the head (under the foot) and a nut trapped in the yoke clamp
collar -> inner race -> spacer -> inner race -> yoke boss, so the yoke and inner races turn
together while the outer races stay in the housing.

Frame: same as base_mount. Shoulder (pitch) joint at the origin, pin along z. The yaw axis is
the x axis (vertical when worn) and passes through the shoulder joint, so yaw and pitch
axes intersect. Everything below the joint sits at negative x. Units mm.
"""
import cadquery as cq
import base_hand as bh

# ---------------- parameters
T, R, PIN_D = bh.T, bh.R, bh.PIN_D
THIN_TO = -40.0                 # thin tongue down to here (keeps L1's swing range)
BLOCK = (-56.0, -40.0)          # solid block under the tongue, holds the M12 nut
BLOCK_HY, BLOCK_HZ = 20.0, 16.0
HUB = (-62.0, -56.0); HUB_D = 50.0
BOSS = (-63.0, -62.0); BOSS_D = 15.0     # bears on the upper bearing's inner race only
HOUSING = (-95.0, -63.0); TUBE_D = 40.0
FOOT = (-95.0, -87.0); FOOT_HY, FOOT_HZ = 35.0, 30.0
FOOT_HOLES = [(25, 20), (25, -20), (-25, 20), (-25, -20)]
BRG_OD, BRG_W, BRG_ID = 28.1, 8.0, 12.0  # 6001: 12 x 28 x 8 (28.1 = printed press fit, test first)
WEB_HOLE = 20.0                 # holds the outer races, clears the inner races
AXLE_HOLE = 12.4
NUT_AF, NUT_H = 19.4, 10.0      # M12 nut 19 mm across flats, 10 mm thick (+ clearance)
NUT_X = (BLOCK[0], BLOCK[0] + NUT_H + 0.4)
M4 = 4.4


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def xcyl(x0, x1, d):
    """Cylinder on the yaw axis (x) from x0 to x1."""
    return cq.Workplane("YZ").workplane(offset=x0).circle(d / 2).extrude(x1 - x0)


# ---------------- shoulder yoke (turns)
yoke = bh.tongue_strip(THIN_TO)
yoke = yoke.union(box(*BLOCK, -BLOCK_HY, BLOCK_HY, -BLOCK_HZ, BLOCK_HZ))
yoke = yoke.union(xcyl(*HUB, HUB_D)).union(xcyl(*BOSS, BOSS_D))
yoke = yoke.cut(cq.Workplane("XY").workplane(offset=-T).circle(PIN_D / 2).extrude(2 * T))   # pitch pin hole
yoke = yoke.cut(xcyl(BOSS[0] - 1, BLOCK[1] - 2, AXLE_HOLE))                                   # axle hole, stops below the tongue
# nut trap: hex pocket around the axle, open to the +z side so the nut slides in
nut = (cq.Workplane("YZ").workplane(offset=NUT_X[0]).polygon(6, NUT_AF / 0.866).extrude(NUT_X[1] - NUT_X[0]))
slot = box(NUT_X[0], NUT_X[1], -NUT_AF / 2, NUT_AF / 2, 0, BLOCK_HZ + 1)
yoke = yoke.cut(nut).cut(slot)

# ---------------- housing (fixed)
housing = xcyl(*HOUSING, TUBE_D).union(box(*FOOT, -FOOT_HY, FOOT_HY, -FOOT_HZ, FOOT_HZ))
for sy, sz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:                     # four ribs, foot to tube
    w = 4.0
    pts = [(FOOT[1], TUBE_D / 2 - 1), (FOOT[1], min(FOOT_HY, FOOT_HZ) - 2), (HOUSING[1] - 8, TUBE_D / 2 - 1)]
    rib = cq.Workplane("XY").workplane(offset=-w / 2).polyline(pts).close().extrude(w)
    if sz:
        rib = rib.rotate((0, 0, 0), (1, 0, 0), 90 * sz)
    elif sy < 0:
        rib = rib.rotate((0, 0, 0), (1, 0, 0), 180)
    housing = housing.union(rib)
housing = housing.cut(xcyl(HOUSING[0] - 1, HOUSING[0] + BRG_W, BRG_OD))      # lower bearing pocket, opens down
housing = housing.cut(xcyl(HOUSING[1] - BRG_W, HOUSING[1] + 1, BRG_OD))      # upper bearing pocket, opens up
housing = housing.cut(xcyl(HOUSING[0], HOUSING[1], WEB_HOLE))
for y, z in FOOT_HOLES:
    housing = housing.cut(cq.Workplane("YZ").workplane(offset=FOOT[0] - 1).center(y, z).circle(M4 / 2).extrude(FOOT[1] - FOOT[0] + 2))

# ---------------- small printed parts
SPACER_LEN = (HOUSING[1] - BRG_W) - (HOUSING[0] + BRG_W)      # gap between the two inner races
yaw_spacer = cq.Workplane("XY").circle(BOSS_D / 2).circle(12.3 / 2).extrude(SPACER_LEN)
yaw_collar = cq.Workplane("XY").circle(BOSS_D / 2).circle(12.3 / 2).extrude(1.5)


def bearing_coupon_28():
    sizes = [27.9, 28.0, 28.1, 28.2]
    p = box(0, 140, 0, 38, 0, 9)
    for i, d in enumerate(sizes):
        cx = 18 + i * 34.5
        p = p.cut(cq.Workplane("XY").workplane(offset=1).center(cx, 19).circle(d / 2).extrude(9))
        p = p.cut(cq.Workplane("XY").workplane(offset=-1).center(cx, 19).circle(10).extrude(3))
        p = p.cut(cq.Workplane("XY").workplane(offset=8.5).center(cx, 3).text(f"{d:.1f}", 3.5, 1, kind="bold"))
    return p


coupon_28 = bearing_coupon_28()
JOINT_ABOVE_FOOT = -HOUSING[0]          # pitch joint height above the foot's underside, mm
