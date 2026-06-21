"""
run_sim.py
----------
Entry point for the FFAI Robothon submission, v2.

Usage (Windows / Mac / Linux, after `pip install -r requirements.txt`):

    python run_sim.py                 # opens an interactive 3D viewer
    python run_sim.py --record demo.mp4 --duration 25   # headless video for submission

Press Ctrl+C in the terminal to stop early.
"""

import argparse
import os
import time

import mujoco
import numpy as np

from controller import FetchController

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "robot_world.xml")

STATE_LABELS = {
    "DRIVE_TO_CUBE": "Driving to cube",
    "LOWER_ARM": "Lowering arm",
    "CLOSE_GRIPPER": "Grasping (real contact grip)",
    "LIFT_ARM": "Lifting (weld constraint engaged)",
    "ALIGN_WRIST": "Re-orienting with wrist joint",
    "DRIVE_TO_GOAL": "Navigating around obstacle",
    "LOWER_AND_OPEN": "Placing on goal pad",
    "DONE": "Done",
}


def run_interactive(model, data, controller):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        done_hold = 0
        while viewer.is_running():
            step_start = time.time()
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()

            if controller.state == "DONE":
                done_hold += 1
                if done_hold > 400:  # keep the result visible for ~2s, then exit
                    break

            # try to roughly keep real-time pace
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


def _caption(frame, text):
    """Draws a simple readable caption in the bottom-left of a frame.
    Falls back to returning the frame unchanged if Pillow isn't installed."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    pad = 8
    box_h = 28
    w, h = img.size
    draw.rectangle([0, h - box_h, w, h], fill=(20, 20, 20))
    draw.text((pad, h - box_h + 6), text, fill=(255, 255, 255))
    return np.array(img)


def run_record(model, data, controller, out_path, duration, fps=30):
    import imageio

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.fixedcamid = model.camera("demo_cam").id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    frames = []
    n_steps = int(duration / model.opt.timestep)
    steps_per_frame = max(int(round(1.0 / (fps * model.opt.timestep))), 1)

    for i in range(n_steps):
        controller.step()
        mujoco.mj_step(model, data)
        if i % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            frame = renderer.render().copy()
            label = STATE_LABELS.get(controller.state, controller.state)
            frame = _caption(frame, f"FetchBot -- {label}")
            frames.append(frame)
        if controller.state == "DONE" and controller.timer > fps * 2:
            break

    renderer.close()
    imageio.mimwrite(out_path, frames, fps=fps, quality=8)
    print(f"[run_sim] wrote {len(frames)} frames -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, default=None,
                        help="If set, render headlessly to this .mp4 path instead of opening a viewer")
    parser.add_argument("--duration", type=float, default=25.0,
                        help="Max seconds of simulated time when recording")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    controller = FetchController(model, data)

    if args.record:
        run_record(model, data, controller, args.record, args.duration)
    else:
        run_interactive(model, data, controller)


if __name__ == "__main__":
    main()
