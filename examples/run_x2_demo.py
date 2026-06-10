from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "assets" / "x2" / "scene.xml"
DEFAULT_OUTPUT = ROOT / "outputs" / "x2_demo.mp4"
DEFAULT_TRAJECTORY = ROOT / "outputs" / "x2_trajectory.json"


BASE_JOINT_POSE = {
    "left_hip_pitch_joint": -0.18,
    "left_hip_roll_joint": 0.08,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.22,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.18,
    "right_hip_roll_joint": -0.08,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.42,
    "right_ankle_pitch_joint": -0.22,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.03,
    "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.75,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.75,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return

    qpos_addr = model.jnt_qposadr[joint_id]
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    progress = smoothstep(0.4, duration_s - 0.8, time_s)
    gait = 2.0 * math.pi * 1.15 * time_s
    wave = math.sin(2.0 * math.pi * 0.7 * time_s)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    # floating_base_joint: x, y, z, qw, qx, qy, qz
    data.qpos[0] = -0.25 + 0.5 * progress
    data.qpos[1] = 0.025 * math.sin(2.0 * math.pi * time_s / max(duration_s, 0.1))
    data.qpos[2] = 0.68 + 0.015 * math.sin(gait)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    pose = dict(BASE_JOINT_POSE)
    pose["waist_yaw_joint"] = 0.12 * math.sin(0.6 * gait)
    pose["head_yaw_joint"] = 0.18 * math.sin(0.5 * gait)
    pose["head_pitch_joint"] = 0.08 * math.sin(0.75 * gait)

    # Small marching motion for the legs. This is a deterministic visualization
    # pose rather than a closed-loop walking controller.
    left_phase = math.sin(gait)
    right_phase = math.sin(gait + math.pi)
    pose["left_hip_pitch_joint"] += 0.10 * left_phase
    pose["left_knee_joint"] += 0.10 * max(0.0, left_phase)
    pose["left_ankle_pitch_joint"] -= 0.05 * left_phase
    pose["right_hip_pitch_joint"] += 0.10 * right_phase
    pose["right_knee_joint"] += 0.10 * max(0.0, right_phase)
    pose["right_ankle_pitch_joint"] -= 0.05 * right_phase

    # Wave the left arm and counter-swing the right arm.
    pose["left_shoulder_pitch_joint"] = 0.15 + 0.35 * wave
    pose["left_shoulder_roll_joint"] = 0.75 + 0.18 * math.sin(1.3 * gait)
    pose["left_elbow_joint"] = -0.85 + 0.28 * math.sin(1.4 * gait)
    pose["left_wrist_yaw_joint"] = 0.45 * math.sin(1.8 * gait)
    pose["right_shoulder_pitch_joint"] = 0.15 - 0.22 * wave
    pose["right_elbow_joint"] = -0.7 - 0.12 * math.sin(1.2 * gait)

    for joint_name, value in pose.items():
        set_joint(model, data, joint_name, value)

    mujoco.mj_forward(model, data)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in MJCF: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def run_demo(
    *,
    model_path: Path,
    video_path: Path,
    trajectory_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        apply_pose(model, data, time_s, duration_s)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

        if frame_idx % max(1, fps // 10) == 0:
            trajectory.append(
                {
                    "time_s": round(time_s, 3),
                    "pelvis_pos": body_position(model, data, "pelvis"),
                    "qpos_head": data.qpos[:10].round(4).tolist(),
                }
            )

    final_pos = body_position(model, data, "pelvis")
    summary = {
        "project": "Agibot X2 MuJoCo Test Demo",
        "task": "The X2 humanoid model loads with its mesh assets and performs a deterministic showcase pose animation.",
        "model": str(model_path),
        "source": "Agibot X2_URDF-v1.3.0 package",
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "duration_s": duration_s,
        "fps": fps,
        "success": final_pos[0] > 0.15,
        "final_pelvis_pos": final_pos,
        "trajectory_samples": trajectory,
    }

    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
    except Exception as exc:
        fallback = video_path.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)

    trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo test video using the Agibot X2 humanoid model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_demo(
        model_path=args.model,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
