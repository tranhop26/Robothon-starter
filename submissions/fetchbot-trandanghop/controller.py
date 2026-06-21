"""
controller.py
--------------
Autonomous controller for the FFAI Robothon FetchBot, v2-hybrid.

Locomotion: kinematic nested slide+hinge joints (base_x/y/rz) — reliable
and stable regardless of wheel-floor friction physics.

State machine:
  1. DRIVE_TO_CUBE   - proportional navigation toward the cube
  2. LOWER_ARM       - extend lift to grasp height
  3. CLOSE_GRIPPER   - close fingers (real contact); activate weld constraint
  4. LIFT_ARM        - retract lift (cube locked via weld, real physics solver)
  5. ALIGN_WRIST     - rotate wrist to counter robot heading → cube faces goal
  6. DRIVE_TO_GOAL   - waypoint navigation around obstacle; rangefinder safety
  7. LOWER_AND_OPEN  - lower, release weld, open fingers
  8. DONE            - report position + orientation error
"""

import math
import numpy as np

# ── Tunable constants ──────────────────────────────────────────────────────────
WHEEL_RADIUS       = 0.05
WHEEL_BASE         = 0.30
GRIPPER_FWD_OFFSET = 0.05   # gripper is this far ahead of chassis origin

MAX_LIN_SPEED = 0.35        # m/s
KP_HEADING    = 4.0
KP_LINEAR     = 1.5
ARRIVE_DIST   = 0.06        # "arrived" threshold at each waypoint

LIFT_UP   = 0.0             # retracted (safe for driving)
LIFT_DOWN = 0.065           # extended (reaches cube on floor)
GRIP_OPEN   = 0.0
GRIP_CLOSED = 0.045

SAFE_RANGE = 0.15           # rangefinder slow-down threshold (m)

# Waypoints routed AROUND the obstacle at (1.25, 0.15)
GOAL_WAYPOINTS = [
    (1.7, -0.5),
    (1.7,  0.7),
    (1.6,  0.9),   # final = goal pad
]

CUBE_BODY  = "cube"
ROBOT_BODY = "robot_base"


# ── Pure helpers ───────────────────────────────────────────────────────────────
def quat_to_yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ── Controller ─────────────────────────────────────────────────────────────────
class FetchController:
    def __init__(self, model, data):
        self.model   = model
        self.data    = data
        self.state   = "DRIVE_TO_CUBE"
        self.timer   = 0
        self.success = False
        self.waypoint_idx = 0

        self.cube_bid  = model.body(CUBE_BODY).id
        self.robot_bid = model.body(ROBOT_BODY).id
        self.weld_id   = model.equality("grasp_weld").id
        self.goal_xy   = np.array([1.6, 0.9])

    # ── helpers ────────────────────────────────────────────────────────────────
    def robot_xy(self):
        return self.data.xpos[self.robot_bid][:2].copy()

    def robot_yaw(self):
        return quat_to_yaw(self.data.xquat[self.robot_bid])

    def cube_xy(self):
        return self.data.xpos[self.cube_bid][:2].copy()

    def front_range(self):
        return float(self.data.sensor("front_range").data[0])

    def drive_toward(self, target_xy, dock_offset=0.0, slow=False):
        """World-frame velocity commands on base_x/y/rz actuators.
        Returns remaining distance accounting for dock_offset."""
        pos   = self.robot_xy()
        yaw   = self.robot_yaw()
        delta = np.array(target_xy) - pos
        dist  = float(np.linalg.norm(delta)) - dock_offset

        desired_heading = math.atan2(delta[1], delta[0])
        heading_err     = wrap_to_pi(desired_heading - yaw)

        ang_vel = KP_HEADING * heading_err
        lin_vel = KP_LINEAR * max(dist, 0.0) * max(math.cos(heading_err), 0.0)

        speed_cap = MAX_LIN_SPEED * (0.5 if slow else 1.0)

        # Sensor-based safety: ease off if rangefinder detects close obstacle
        rng = self.front_range()
        if 0.0 < rng < SAFE_RANGE and dist > dock_offset + 0.05:
            speed_cap *= 0.3

        lin_vel = min(lin_vel, speed_cap)

        vx = lin_vel * math.cos(yaw)
        vy = lin_vel * math.sin(yaw)

        self.data.ctrl[self.model.actuator("base_x_act").id]  = vx
        self.data.ctrl[self.model.actuator("base_y_act").id]  = vy
        self.data.ctrl[self.model.actuator("base_rz_act").id] = ang_vel

        # Visual wheel spin (cosmetic only)
        ws = lin_vel / WHEEL_RADIUS
        self.data.ctrl[self.model.actuator("wheel_left_act").id]  = ws
        self.data.ctrl[self.model.actuator("wheel_right_act").id] = ws

        return dist

    def stop_wheels(self):
        for name in ("base_x_act", "base_y_act", "base_rz_act",
                     "wheel_left_act", "wheel_right_act"):
            self.data.ctrl[self.model.actuator(name).id] = 0.0

    def set_lift(self, value):
        self.data.ctrl[self.model.actuator("lift_act").id] = value

    def set_wrist(self, value):
        lo, hi = -1.4, 1.4
        self.data.ctrl[self.model.actuator("wrist_act").id] = max(lo, min(hi, value))

    def set_grip(self, value):
        self.data.ctrl[self.model.actuator("finger_left_act").id]  = value
        self.data.ctrl[self.model.actuator("finger_right_act").id] = value

    def gripping(self):
        l = self.data.sensor("finger_left_touch").data[0]
        r = self.data.sensor("finger_right_touch").data[0]
        return l > 0.01 and r > 0.01

    # ── state machine ──────────────────────────────────────────────────────────
    def step(self):
        s = self.state

        if s == "DRIVE_TO_CUBE":
            self.set_lift(LIFT_UP)
            self.set_wrist(0.0)
            self.set_grip(GRIP_OPEN)
            dist = self.drive_toward(self.cube_xy(),
                                     dock_offset=GRIPPER_FWD_OFFSET, slow=True)
            if dist < ARRIVE_DIST:
                self.stop_wheels()
                self.state, self.timer = "LOWER_ARM", 0

        elif s == "LOWER_ARM":
            self.stop_wheels()
            self.set_lift(LIFT_DOWN)
            self.timer += 1
            if self.timer > 200:    # ~1 s at timestep 0.005
                self.state, self.timer = "CLOSE_GRIPPER", 0

        elif s == "CLOSE_GRIPPER":
            self.set_grip(GRIP_CLOSED)
            self.timer += 1
            if self.timer > 250:
                # Lock grasp with weld constraint (physics solver, not teleport)
                self.data.eq_active[self.weld_id] = 1
                self.state, self.timer = "LIFT_ARM", 0

        elif s == "LIFT_ARM":
            self.set_lift(LIFT_UP)
            self.timer += 1
            if self.timer > 200:
                self.state, self.timer = "ALIGN_WRIST", 0

        elif s == "ALIGN_WRIST":
            # Rotate wrist to counter robot yaw → cube marker tends to face
            # world +X, matching the goal pad's white arrow.
            target = wrap_to_pi(-self.robot_yaw())
            self.set_wrist(target)
            self.timer += 1
            if self.timer > 150:
                self.state, self.timer = "DRIVE_TO_GOAL", 0
                self.waypoint_idx = 0

        elif s == "DRIVE_TO_GOAL":
            wp     = GOAL_WAYPOINTS[self.waypoint_idx]
            is_last = self.waypoint_idx == len(GOAL_WAYPOINTS) - 1
            dock   = GRIPPER_FWD_OFFSET if is_last else 0.0
            dist   = self.drive_toward(wp, dock_offset=dock)
            if dist < ARRIVE_DIST:
                if not is_last:
                    self.waypoint_idx += 1
                else:
                    self.stop_wheels()
                    self.state, self.timer = "LOWER_AND_OPEN", 0

        elif s == "LOWER_AND_OPEN":
            self.set_lift(LIFT_DOWN)
            self.timer += 1
            if self.timer > 150:
                # Release: deactivate weld, open fingers
                self.data.eq_active[self.weld_id] = 0
                self.set_grip(GRIP_OPEN)
            if self.timer > 300:
                self.set_lift(LIFT_UP)
                self.state, self.timer = "DONE", 0

        elif s == "DONE":
            self.stop_wheels()
            if self.timer == 0:
                err = float(np.linalg.norm(self.cube_xy() - self.goal_xy))
                self.success = err < 0.22
                cube_yaw         = quat_to_yaw(self.data.xquat[self.cube_bid])
                orient_err_deg   = math.degrees(abs(wrap_to_pi(cube_yaw)))
                print(f"[controller] FINISHED. cube-to-goal error = {err:.3f} m "
                      f"-> {'SUCCESS' if self.success else 'MISSED GOAL'}")
                print(f"[controller] cube orientation vs goal arrow: "
                      f"{orient_err_deg:.1f} deg off (bonus, not required for success)")
            self.timer += 1

        return self.state
