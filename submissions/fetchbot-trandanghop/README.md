# FetchBot — FFAI Robothon Summer 2026 Submission

A minimal mobile-manipulation robot built with **MuJoCo 3**.  
The robot drives to a cube, grasps it, navigates around a static obstacle, and
deposits the cube on the green goal pad — all autonomously, using a classic
state-machine controller (no ML required).

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify MuJoCo loaded correctly
python -c "import mujoco; print(mujoco.__version__)"

# 3. Load-test the XML scene
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('model/robot_world.xml'); print('OK', m.nq, m.nu)"

# 4. Run interactive simulation
python run_sim.py

# 5. Record submission video
python run_sim.py --record demo.mp4 --duration 25
```

---

## Project structure

```
robothon-submission/
├── model/robot_world.xml   # MuJoCo MJCF scene (robot + environment)
├── controller.py           # Autonomous state-machine controller
├── run_sim.py              # Entry point (viewer / --record mode)
├── requirements.txt
└── README.md
```

---

## How it works

The controller (`controller.py`) moves through these states in order:

| # | State | What happens |
|---|-------|-------------|
| 1 | `DRIVE_TO_CUBE` | Proportional navigation toward the orange cube |
| 2 | `LOWER_ARM` | Extend the vertical lift down to grasp height |
| 3 | `CLOSE_GRIPPER` | Close fingers around the cube |
| 4 | `LIFT_ARM` | Retract lift so cube clears the floor |
| 5 | `DRIVE_TO_GOAL` | Follow waypoints that route **around** the red obstacle |
| 6 | `LOWER_AND_OPEN` | Lower lift and release cube on the green pad |
| 7 | `DONE` | Stop; print `SUCCESS` or `MISSED GOAL` |

---

## If something looks slightly off

| Symptom | What to tune | Where |
|---------|-------------|-------|
| Gripper misses cube vertically | `LIFT_DOWN` (range 0–0.09, try ±0.01) | `controller.py` |
| Cube slips during transport | `GRIP_CLOSED` (max 0.045) or `friction` values | `controller.py` / XML |
| Robot wiggles/oscillates | Lower `KP_HEADING` (e.g. 4.0 → 2.5) | `controller.py` |
| Robot overshoots waypoints | Raise `KP_HEADING` or lower `MAX_LIN_SPEED` | `controller.py` |
| Robot drives into obstacle | Push waypoints further out in `GOAL_WAYPOINTS` | `controller.py` |

---

## Ideas for extension

- Replace ground-truth position reads with camera/depth perception
- Add obstacle avoidance using the `front_range` sensor
- Train a policy with MuJoCo's built-in RL hooks

---

*Submitted to FFAI Robothon Summer 2026.*
