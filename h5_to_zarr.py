#!/usr/bin/env python3
"""
h5_to_zarr.py

Convert an imitation-learning HDF5 dataset with structure:

/
|-- arm_joint_names
|-- data/
|   |-- demo_0/
|   |   |-- actions  (T, 10) float32     # Δpos(3) + rot6D(6) + gripper(1)
|   |   |-- obs/
|   |   |   |-- arm_q     (T, 7)
|   |   |   |-- eef_pos   (T, 3)
|   |   |   |-- eef_quat  (T, 4)         # quaternion per timestep
|   |   |   |-- gripper   (T, 1)
|   |   |   |-- images_0  (T, H, W, 3) uint8
|   |   |   |-- images_1  (T, H, W, 3) uint8  (optional)
|   |   |   |-- ...

into a Diffusion-Policy-friendly Zarr replay buffer with:

OUTPUT_ZARR/
  data/
    action               (sumT, 10) float32
    lowdim_joint_states  (sumT, 7) float32   # arm_q
    lowdim_ee_states     (sumT, 10) float32  # [eef_pos(3), eef_rot6d(6), gripper(1)]
    rgb_<key>            (sumT, H, W, 3) uint8    # one array per camera, e.g. rgb_images_0, rgb_images_1
  meta/
    episode_ends  (num_demos,) int64        # cumulative end indices (exclusive)

Notes:
- We convert eef_quat -> eef_rot6d (absolute orientation) for a continuous observation feature:
    rot6d = [R[:,0], R[:,1]] where R is rotation matrix from unit quaternion.
- Your ACTIONS are left untouched (already Δpos + rot6D + gripper), suitable for diffusion policy.
- Multi-camera support: set RGB_KEYS = ["images_0", "images_1", ...]
- Quaternion order: by default expects (x, y, z, w). If your file stores (w, x, y, z),
  set QUAT_WXYZ = True to reorder.

Usage examples:
  python h5_to_zarr.py data/augmented_sweep.h5
  python h5_to_zarr.py /path/to/demo.h5 -o /path/to/demo.zarr
"""

import argparse
import os
import shutil
import h5py
import numpy as np
import zarr
from numcodecs import Blosc

# Choose which camera streams to export. You can later select them in YAML via rgb_keys.
RGB_KEYS = ["images_0", "images_1", "images_2"]  # e.g. ["images_0"] for single cam

# Quaternion storage convention in H5:
# False -> eef_quat is (x,y,z,w)
# True  -> eef_quat is (w,x,y,z) and will be reordered to (x,y,z,w)
QUAT_WXYZ = False

OVERWRITE = True
VERIFY = True

# Compression (good defaults)
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _sorted_demo_keys(h5_data_group):
    keys = list(h5_data_group.keys())
    def key_fn(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return k
    return sorted(keys, key=key_fn)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit length."""
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) -> rotation matrix.
    q_xyzw: (..., 4) float
    returns: (..., 3, 3) float32
    """
    q = normalize_quat(q_xyzw).astype(np.float32)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R[..., 0, 0] = 1.0 - 2.0*(yy + zz)
    R[..., 0, 1] = 2.0*(xy - wz)
    R[..., 0, 2] = 2.0*(xz + wy)

    R[..., 1, 0] = 2.0*(xy + wz)
    R[..., 1, 1] = 1.0 - 2.0*(xx + zz)
    R[..., 1, 2] = 2.0*(yz - wx)

    R[..., 2, 0] = 2.0*(xz - wy)
    R[..., 2, 1] = 2.0*(yz + wx)
    R[..., 2, 2] = 1.0 - 2.0*(xx + yy)

    return R


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    Rotation 6D (Zhou et al.) from rotation matrix:
    take first two columns and concatenate -> (..., 6)
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1).astype(np.float32)


def ensure_overwrite(path: str, overwrite: bool):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"{path} exists. Set OVERWRITE=True to replace it.")


def _default_output_path(input_path: str) -> str:
    base = os.path.splitext(input_path)[0]
    return base + ".zarr"


def main(input_h5: str, output_zarr: str):
    ensure_overwrite(output_zarr, OVERWRITE)
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)

    # We accumulate per-demo arrays then concatenate (robust + simple).
    # If your dataset is extremely large, we can rewrite this to stream.
    all_actions = []
    all_lowdim_joint = []
    all_lowdim_ee = []
    all_rgbs = {k: [] for k in RGB_KEYS}
    episode_ends = []

    total_T = 0

    with h5py.File(input_h5, "r") as f:
        if "data" not in f:
            raise KeyError("Expected root group 'data' not found in H5.")

        demo_keys = _sorted_demo_keys(f["data"])
        if len(demo_keys) == 0:
            raise ValueError("No demos found under /data in H5.")

        # Validate RGB keys exist in first demo
        obs0 = f[f"data/{demo_keys[0]}/obs"]
        for k in RGB_KEYS:
            if k not in obs0:
                raise KeyError(
                    f"RGB key '{k}' not found in /data/{demo_keys[0]}/obs. "
                    f"Available: {list(obs0.keys())}"
                )

        for dname in demo_keys:
            dpath = f"data/{dname}"
            obs = f[f"{dpath}/obs"]

            actions = f[f"{dpath}/actions"][...].astype(np.float32)  # (T,10)
            if actions.ndim != 2 or actions.shape[1] != 10:
                raise ValueError(f"{dpath}/actions expected (T,10), got {actions.shape}")
            T = actions.shape[0]

            arm_q = obs["arm_q"][...].astype(np.float32)       # (T,7)
            eef_pos = obs["eef_pos"][...].astype(np.float32)   # (T,3)
            eef_quat = obs["eef_quat"][...].astype(np.float32) # (T,4)
            gripper = obs["gripper"][...].astype(np.float32)   # (T,1)

            # Basic length checks
            for name, arr in [("arm_q", arm_q), ("eef_pos", eef_pos), ("eef_quat", eef_quat), ("gripper", gripper)]:
                if arr.shape[0] != T:
                    raise ValueError(f"Length mismatch in {dname}: {name} T={arr.shape[0]} vs actions T={T}")

            # Load selected camera streams
            rgb_this = {}
            for k in RGB_KEYS:
                rgb = obs[k][...]
                if rgb.ndim != 4 or rgb.shape[-1] != 3:
                    raise ValueError(f"{dpath}/obs/{k} expected (T,H,W,3), got {rgb.shape}")
                if rgb.shape[0] != T:
                    raise ValueError(f"Length mismatch in {dname}: {k} T={rgb.shape[0]} vs actions T={T}")
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                rgb_this[k] = rgb

            # Quaternion reorder if needed
            # Stored as (w,x,y,z) -> convert to (x,y,z,w)
            if QUAT_WXYZ:
                eef_quat = eef_quat[..., [1, 2, 3, 0]]

            # Convert eef_quat -> absolute eef_rot6d (continuous orientation feature)
            R = quat_xyzw_to_rotmat(eef_quat)       # (T,3,3)
            eef_rot6d = rotmat_to_rot6d(R)          # (T,6)

            # lowdim split for easier config control
            lowdim_joint = arm_q.astype(np.float32)  # (T,7)
            lowdim_ee = np.concatenate([eef_pos, eef_rot6d, gripper], axis=-1).astype(np.float32)  # (T,10)
            if lowdim_joint.shape != (T, 7):
                raise RuntimeError(f"{dname}: lowdim_joint expected (T,7), got {lowdim_joint.shape}")
            if lowdim_ee.shape != (T, 10):
                raise RuntimeError(f"{dname}: lowdim_ee expected (T,10), got {lowdim_ee.shape}")

            all_actions.append(actions)
            all_lowdim_joint.append(lowdim_joint)
            all_lowdim_ee.append(lowdim_ee)
            for k in RGB_KEYS:
                all_rgbs[k].append(rgb_this[k])

            total_T += T
            episode_ends.append(total_T)

            print(f"[OK] {dname}: T={T} | joint_angles=7, ee_states=10 | cams={RGB_KEYS}")

    # Concatenate across demos
    action = np.concatenate(all_actions, axis=0)           # (sumT, 10)
    lowdim_joint = np.concatenate(all_lowdim_joint, axis=0)  # (sumT, 7)
    lowdim_ee = np.concatenate(all_lowdim_ee, axis=0)        # (sumT, 10)
    rgbs = {k: np.concatenate(all_rgbs[k], axis=0) for k in RGB_KEYS}  # each (sumT,H,W,3)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    # Write Zarr
    root = zarr.open(output_zarr, mode="w")
    g_data = root.create_group("data")
    g_meta = root.create_group("meta")

    g_data.create_dataset(
        "action",
        data=action,
        dtype=np.float32,
        chunks=(min(4096, action.shape[0]), action.shape[1]),
        compressor=COMPRESSOR
    )
    g_data.create_dataset(
        "lowdim_joint_states",
        data=lowdim_joint,
        dtype=np.float32,
        chunks=(min(4096, lowdim_joint.shape[0]), lowdim_joint.shape[1]),
        compressor=COMPRESSOR
    )
    g_data.create_dataset(
        "lowdim_ee_states",
        data=lowdim_ee,
        dtype=np.float32,
        chunks=(min(4096, lowdim_ee.shape[0]), lowdim_ee.shape[1]),
        compressor=COMPRESSOR
    )

    # One dataset per camera: data/rgb_images_0, data/rgb_images_1, ...
    for k, rgb in rgbs.items():
        g_data.create_dataset(
            f"rgb_{k}",
            data=rgb,
            dtype=np.uint8,
            chunks=(min(256, rgb.shape[0]), rgb.shape[1], rgb.shape[2], rgb.shape[3]),
            compressor=COMPRESSOR
        )

    g_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype=np.int64,
        chunks=(min(1024, episode_ends.shape[0]),),
        compressor=COMPRESSOR
    )

    print("\n=== WROTE ZARR REPLAY ===")
    print("Path:", output_zarr)
    print("data/action :", action.shape, action.dtype)
    print("data/lowdim_joint_states :", lowdim_joint.shape, lowdim_joint.dtype, "(arm_q7)")
    print("data/lowdim_ee_states    :", lowdim_ee.shape, lowdim_ee.dtype, "(eef_pos3 + eef_rot6d6 + gripper1)")
    for k in RGB_KEYS:
        rgb = rgbs[k]
        print(f"data/rgb_{k}:", rgb.shape, rgb.dtype)
    print("meta/episode_ends:", episode_ends.shape, episode_ends.dtype, "last:", episode_ends[-1])

    if VERIFY:
        r = zarr.open(output_zarr, mode="r")
        T = r["data"]["action"].shape[0]
        assert r["data"]["lowdim_joint_states"].shape == (T, 7)
        assert r["data"]["lowdim_ee_states"].shape == (T, 10)
        for k in RGB_KEYS:
            assert r["data"][f"rgb_{k}"].shape[0] == T
            assert r["data"][f"rgb_{k}"].shape[-1] == 3
        assert r["meta"]["episode_ends"][-1] == T
        print("[VERIFY] Passed basic integrity checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an HDF5 dataset to a Diffusion-Policy Zarr replay.")
    parser.add_argument("input_h5", help="Path to input .h5 file.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output .zarr (default: same folder/name as input)."
    )
    args = parser.parse_args()

    input_h5 = args.input_h5
    output_zarr = args.output if args.output is not None else _default_output_path(input_h5)

    print(f"[INFO] Input H5: {input_h5}")
    print(f"[INFO] Output Zarr: {output_zarr}")
    main(input_h5, output_zarr)
