#!/usr/bin/env python3
"""
Rollout a trained diffusion policy on Spot.

Usage example:
python rollout/spot_rollout.py \\
  --ckpt data/outputs/.../checkpoints/latest.ckpt \\
  --image-map images_0=frontleft_fisheye_image \\
  --image-map images_2=hand_color_image \\
  --control-hz 10 \\
  --num-inference-steps 16

Safety: keep a hand near the robot E-stop. This script sends live commands.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import dill
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _add_spot_repo_to_path(spot_repo: str):
    if spot_repo not in sys.path:
        sys.path.append(spot_repo)


def _parse_kv_list(items: List[str]) -> Dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value for --image-map, got '{item}'")
        key, val = item.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def _get_shape_meta(cfg) -> dict:
    if "policy" in cfg and "shape_meta" in cfg.policy:
        return OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    if "task" in cfg and "shape_meta" in cfg.task:
        return OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    raise KeyError("shape_meta not found in cfg.policy or cfg.task.")


def _extract_obs_keys(shape_meta: dict) -> Tuple[List[str], List[str], Dict[str, Tuple[int, int]]]:
    obs_meta = shape_meta["obs"]
    rgb_keys = []
    lowdim_keys = []
    img_shapes = {}
    for key, attr in obs_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
            shape = attr["shape"]
            if len(shape) != 3:
                raise ValueError(f"Expected image shape [C,H,W] for {key}, got {shape}")
            img_shapes[key] = (int(shape[1]), int(shape[2]))  # (H, W)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys, img_shapes


def _build_joint_order(state, joint_names):
    if joint_names is not None:
        return joint_names
    names = []
    for j in state.kinematic_state.joint_states:
        if j.name.startswith("arm0."):
            names.append(j.name)
    if len(names) == 0:
        raise RuntimeError("No arm0.* joints found in robot state.")
    return names


def _extract_joint_states(state, joint_names):
    name_to_pos = {}
    for j in state.kinematic_state.joint_states:
        if j.name.startswith("arm0."):
            name_to_pos[j.name] = j.position.value
    q = [name_to_pos[n] for n in joint_names]
    return np.array(q, dtype=np.float32)


def _quat_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    R = np.array([
        [1 - ty * y - tz * z, tx * y - tz * w,     tx * z + ty * w],
        [tx * y + tz * w,     1 - tx * x - tz * z, ty * z - tx * w],
        [tx * z - ty * w,     ty * z + tx * w,     1 - tx * x - ty * y]
    ], dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def _resize_bgr(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if img.shape[:2] == (th, tw):
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--device", default="cuda:0", help="Torch device (e.g., cuda:0 or cpu)")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Control frequency in Hz")
    parser.add_argument("--steps-per-inference", type=int, default=1, help="How many actions to execute per policy call")
    parser.add_argument("--num-inference-steps", type=int, default=16, help="Diffusion sampling steps")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Max rollout duration in seconds")
    parser.add_argument("--image-map", action="append", default=[], help="Map obs key to Spot image source (key=source)")
    parser.add_argument("--spot-repo", default=os.environ.get("SPOT_META_TELEOP_DIR", "/home/akash/UNH/spot/spot-meta-teleop"))
    parser.add_argument("--undock", action="store_true", help="Undock before starting")
    parser.add_argument("--reset-pose", action="store_true", help="Move arm to default pose before starting")
    parser.add_argument("--dock", action="store_true", help="Dock after rollout")
    parser.add_argument("--show", action="store_true", help="Show camera feed window")
    args = parser.parse_args()

    if not os.path.isdir(args.spot_repo):
        raise FileNotFoundError(f"Spot repo not found: {args.spot_repo}")
    _add_spot_repo_to_path(args.spot_repo)

    from spot_controller import SpotRobotController
    from utils.spot_utils import image_to_cv
    from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME

    # Load checkpoint + policy
    payload = torch.load(open(args.ckpt, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval().to(args.device)
    policy.num_inference_steps = args.num_inference_steps

    shape_meta = _get_shape_meta(cfg)
    rgb_keys, lowdim_keys, img_shapes = _extract_obs_keys(shape_meta)
    if len(rgb_keys) == 0:
        raise RuntimeError("No rgb keys found in shape_meta.")

    # Build image source map
    image_map = _parse_kv_list(args.image_map)
    default_map = {
        "images_2": "frontleft_fisheye_image",
        "images_1": "frontright_fisheye_image",
        "images_0": "hand_color_image",
    }
    for key in rgb_keys:
        if key not in image_map:
            if key in default_map:
                image_map[key] = default_map[key]
            else:
                raise KeyError(f"Missing --image-map for rgb key '{key}'")

    # Spot controller
    robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.1.138")
    user = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")
    controller = SpotRobotController(robot_ip, user, password, default_exec_time=0.3)

    if args.undock:
        controller.undock()
    if args.reset_pose:
        controller.reset_pose()

    # rolling buffers for obs history
    n_obs_steps = int(cfg.n_obs_steps)
    history = {k: deque(maxlen=n_obs_steps) for k in (rgb_keys + lowdim_keys)}
    joint_names = None

    def capture_obs():
        nonlocal joint_names
        state = controller.current_state()
        joint_names = _build_joint_order(state, joint_names)
        joint_states = _extract_joint_states(state, joint_names)
        if joint_states.shape[0] != 7:
            raise RuntimeError(f"Expected 7 joint states, got {joint_states.shape[0]}")

        snap = state.kinematic_state.transforms_snapshot
        pose = get_a_tform_b(snap, BODY_FRAME_NAME, "hand")
        pos = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        quat = np.array([pose.rot.x, pose.rot.y, pose.rot.z, pose.rot.w], dtype=np.float32)
        rot6d = _quat_to_rot6d(quat)
        gripper = float(state.manipulator_state.gripper_open_percentage) / 100.0
        ee_states = np.concatenate([pos, rot6d, np.array([gripper], dtype=np.float32)], axis=0)

        for key in rgb_keys:
            src = image_map[key]
            resp = controller.spot_images.get_rgb_image(src)
            if resp is None:
                raise RuntimeError(f"Failed to fetch image from source '{src}'")
            img = image_to_cv(resp)  # BGR uint8
            img = _resize_bgr(img, img_shapes[key])
            history[key].append(img)

        for key in lowdim_keys:
            if key == "joint_states":
                history[key].append(joint_states)
            elif key == "ee_states":
                history[key].append(ee_states)
            else:
                raise KeyError(f"Unsupported lowdim key '{key}' in rollout.")

        if args.show:
            vis_key = rgb_keys[0]
            cv2.imshow("spot_rollout", history[vis_key][-1])
            cv2.waitKey(1)

    # Warm-up history
    for _ in range(n_obs_steps):
        capture_obs()
        time.sleep(1.0 / args.control_hz)

    plan = None
    plan_idx = 0
    steps_per_inference = max(1, int(args.steps_per_inference))

    dt = 1.0 / args.control_hz
    t_start = time.monotonic()
    t_next = t_start

    with torch.no_grad():
        if hasattr(policy, "reset"):
            policy.reset()
        while (time.monotonic() - t_start) < args.max_seconds:
            capture_obs()

            if (plan is None) or (plan_idx >= steps_per_inference) or (plan_idx >= plan.shape[0]):
                obs = {}
                for key in rgb_keys:
                    frames = np.stack(history[key], axis=0).astype(np.float32) / 255.0
                    frames = np.moveaxis(frames, -1, 1)  # T,C,H,W
                    obs[key] = torch.from_numpy(frames)
                for key in lowdim_keys:
                    vals = np.stack(history[key], axis=0).astype(np.float32)
                    obs[key] = torch.from_numpy(vals)
                obs = dict_apply(obs, lambda x: x.unsqueeze(0).to(args.device))

                result = policy.predict_action(obs)
                plan = result["action"][0].detach().to("cpu").numpy()
                plan_idx = 0

            action = plan[plan_idx]
            plan_idx += 1
            controller.apply_action(action)

            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))

    if args.dock:
        controller.stow_arm()
        controller.dock()


if __name__ == "__main__":
    main()
