from __future__ import annotations

import argparse
import json
import math
import re
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
DEFAULT_URDF = ROOT / "assets" / "zsl-1" / "urdf" / "ZSL-1_mujoco.urdf"
DEFAULT_OUTPUT = ROOT / "outputs" / "zsl1_demo_v4_wide.mp4"
DEFAULT_TRAJECTORY = ROOT / "outputs" / "zsl1_trajectory_v4_wide.json"

LEGS = ("FL", "FR", "RR", "RL")
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def ensure_mujoco_urdf(source_urdf: Path, output_urdf: Path) -> Path:
    if output_urdf.exists():
        return output_urdf

    text = source_urdf.read_text(encoding="utf-8")
    text = re.sub(r'filename="\.\./meshes/([^"]+)"', r'filename="\1"', text)
    if "<mujoco>" not in text:
        text = text.replace(
            '<robot\n  name="ZSL-1">',
            '<robot\n  name="ZSL-1">\n  <mujoco>\n'
            '    <compiler meshdir="../meshes" discardvisual="false"/>\n'
            "  </mujoco>\n",
        )
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(text, encoding="utf-8")
    return output_urdf


def build_model(urdf_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]

    base = spec.body("BASE_LINK")
    if base is None:
        raise ValueError("Missing BASE_LINK body in ZSL-1 URDF")
    base.add_freejoint(name="floating_base_joint")

    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.06, 0.07, 0.08, 1],
    )
    world.add_geom(
        name="runway",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.05, 0, 0.002],
        size=[0.62, 0.24, 0.002],
        rgba=[0.10, 0.13, 0.16, 1],
    )
    world.add_geom(
        name="start_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.35, 0, 0.006],
        size=[0.08, 0.20, 0.003],
        rgba=[0.15, 0.38, 1.00, 0.75],
    )
    world.add_geom(
        name="goal_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.40, 0, 0.007],
        size=[0.08, 0.20, 0.004],
        rgba=[0.25, 1.00, 0.45, 0.80],
    )
    world.add_light(pos=[0, -1.1, 2.4], dir=[0, 0.35, -1], diffuse=[1.0, 1.0, 1.0])
    world.add_light(pos=[-1.0, 0.8, 1.5], dir=[0.4, -0.3, -1], diffuse=[0.5, 0.55, 0.65])
    world.add_camera(
        name="demo_camera",
        pos=[1.35, -1.05, 0.42],
        xyaxes=[0.9, 0.44, 0, -0.12, 0.24, 0.96],
    )
    return spec.compile()


def style_model_for_video(model: mujoco.MjModel) -> None:
    """Make the SolidWorks URDF import easier to read in a short video."""
    body_shell = np.array([0.92, 0.95, 1.00, 1.0], dtype=np.float32)
    hip_shell = np.array([1.00, 0.52, 0.12, 1.0], dtype=np.float32)
    leg_shell = np.array([0.18, 0.22, 0.28, 1.0], dtype=np.float32)
    foot_shell = np.array([0.05, 0.06, 0.07, 1.0], dtype=np.float32)

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name in {"floor", "runway", "start_pad", "goal_pad"}:
            continue

        # The imported URDF has duplicate visual/collision meshes. Keep the
        # visual group and hide the collision group so the silhouette is clean.
        if model.geom_group[geom_id] == 0:
            model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
            continue

        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or ""
        if body_name == "BASE_LINK":
            model.geom_rgba[geom_id] = body_shell
        elif "ABAD" in body_name or "HIP" in body_name:
            model.geom_rgba[geom_id] = hip_shell
        elif "FOOT" in body_name:
            model.geom_rgba[geom_id] = foot_shell
        else:
            model.geom_rgba[geom_id] = leg_shell


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int | None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id])


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    qpos_addr = joint_qpos_addr(model, joint_name)
    if qpos_addr is None:
        return
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    progress = smoothstep(0.4, duration_s - 0.8, time_s)
    gait = 2.0 * math.pi * 1.55 * time_s
    settle = smoothstep(0.0, 0.7, time_s)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0] = -0.42 + 0.92 * progress
    data.qpos[1] = 0.05 * math.sin(2.0 * math.pi * time_s / max(duration_s, 0.1))
    data.qpos[2] = 0.33 + settle * 0.02 * math.sin(gait)
    yaw = 0.10 * settle * math.sin(2.0 * math.pi * time_s / max(duration_s, 0.1))
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    for leg in LEGS:
        phase = LEG_PHASE[leg]
        swing = math.sin(gait + phase)
        lift = max(0.0, swing)
        set_joint(model, data, f"{leg}_ABAD_JOINT", settle * 0.10 * math.sin(gait + phase + 0.4))
        set_joint(model, data, f"{leg}_HIP_JOINT", 0.58 + settle * 0.32 * swing)
        set_joint(model, data, f"{leg}_KNEE_JOINT", -1.08 + settle * 0.34 * lift)

    mujoco.mj_forward(model, data)


def update_follow_camera(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, time_s: float) -> None:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE_LINK")
    lookat = data.xpos[base_id].copy() if base_id >= 0 else np.array([0.0, 0.0, 0.25])
    lookat[2] = 0.17

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat

    # Wider, steadier shot for reviewing the full robot silhouette and path.
    if time_s < 2.1:
        camera.distance = 1.15
        camera.azimuth = 98.0 + 3.0 * math.sin(1.1 * time_s)
        camera.elevation = -18.0
    elif time_s < 4.2:
        blend = smoothstep(2.1, 4.2, time_s)
        camera.distance = 1.25 - 0.10 * blend
        camera.azimuth = 122.0 + 22.0 * blend
        camera.elevation = -19.0 + 2.0 * blend
    else:
        camera.distance = 1.10
        camera.azimuth = 178.0 + 4.0 * math.sin(1.0 * time_s)
        camera.elevation = -17.0


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in model: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def run_demo(
    *,
    urdf_path: Path,
    video_path: Path,
    trajectory_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
) -> dict:
    source_urdf = urdf_path
    if urdf_path.name == "ZSL-1_mujoco.urdf" and not urdf_path.exists():
        source_urdf = urdf_path.with_name("ZSL-1.urdf")
    if source_urdf.name == "ZSL-1.urdf":
        urdf_path = ensure_mujoco_urdf(source_urdf, urdf_path)

    model = build_model(urdf_path)
    style_model_for_video(model)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        apply_pose(model, data, time_s, duration_s)
        update_follow_camera(model, data, camera, time_s)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

        if frame_idx % max(1, fps // 10) == 0:
            trajectory.append(
                {
                    "time_s": round(time_s, 3),
                    "base_pos": body_position(model, data, "BASE_LINK"),
                }
            )

    final_pos = body_position(model, data, "BASE_LINK")
    summary = {
        "project": "ZSL-1 Robot Dog MuJoCo Test Demo",
        "task": "The packaged ZSL-1 quadruped URDF loads with STL meshes and performs a deterministic patrol gait animation.",
        "model": str(urdf_path),
        "source": str(ROOT / "assets" / "zsl-1"),
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "duration_s": duration_s,
        "fps": fps,
        "success": final_pos[0] > 0.2,
        "final_base_pos": final_pos,
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
        description="Generate a MuJoCo demo video using the packaged ZSL-1 robot dog URDF."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_demo(
        urdf_path=args.urdf,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
