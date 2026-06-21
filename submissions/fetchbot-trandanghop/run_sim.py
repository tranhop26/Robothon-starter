"""
run_sim.py
----------
Entry point for the FFAI Robothon FetchBot simulation.

Usage:
  python run_sim.py                          # interactive viewer
  python run_sim.py --record demo.mp4        # record a video (headless)
  python run_sim.py --record demo.mp4 --duration 30
"""

import argparse
import os
import sys

import mujoco
import numpy as np

from controller import FetchController

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "robot_world.xml")


def run_interactive(model, data):
    """Launch the MuJoCo passive viewer and step the sim in a callback."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        ctrl = FetchController(model, data)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            ctrl.step()
            viewer.sync()
            if ctrl.state == "DONE" and ctrl.timer > 60:
                break


def run_record(model, data, output_path: str, duration: float):
    """Step the sim headlessly and write frames to a video file."""
    try:
        import imageio
    except ImportError:
        sys.exit("imageio is required for recording. Run: pip install imageio[ffmpeg]")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")

    ctrl = FetchController(model, data)
    fps = 30
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    total_steps = int(duration / model.opt.timestep)

    frames = []
    step_idx = 0
    print(f"Recording {duration}s of simulation -> {output_path}")
    while step_idx < total_steps:
        mujoco.mj_step(model, data)
        ctrl.step()
        if step_idx % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())
            print(f"\r  simtime {data.time:.1f}/{duration:.1f}s  state={ctrl.state}   ", end="")
        step_idx += 1
        if ctrl.state == "DONE" and ctrl.timer > 60:
            # Grab a few more seconds so the final state is visible
            for _ in range(fps * 3):
                mujoco.mj_step(model, data)
                ctrl.step()
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render())
            break

    print(f"\nWriting {len(frames)} frames …")
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264",
                     output_params=["-crf", "22", "-preset", "fast"])
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FetchBot simulation runner")
    parser.add_argument("--record", metavar="OUTPUT.mp4", default=None,
                        help="Record a video instead of opening the interactive viewer")
    parser.add_argument("--duration", type=float, default=25.0,
                        help="Simulated seconds to record (default: 25)")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if args.record:
        run_record(model, data, args.record, args.duration)
    else:
        run_interactive(model, data)


if __name__ == "__main__":
    main()
