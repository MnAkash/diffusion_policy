from typing import Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.sampler import create_indices, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class SpotZarrImageDataset(BaseImageDataset):
    """
    Zarr layout expected (from h5_to_zarr.py):

      data/action                 (T, 10) float32
      data/lowdim_<key>           (T, D) float32
          e.g. data/lowdim_joint_states (T, 7), data/lowdim_ee_states (T, 10)

      data/rgb_<key>              (T, H, W, 3) uint8
          e.g. data/rgb_images_0, data/rgb_images_1, ...

      meta/episode_ends           (E,) int64 cumulative ends (exclusive)

    YAML-selectable cameras:
      rgb_keys: ["images_0"] or ["images_0", "images_1"]

    Returns one image tensor per camera key:
      obs['images_0']: (To, 3, H, W) float32 in [0,1]
      obs['images_1']: (To, 3, H, W) float32 in [0,1]
      obs['joint_states']: (To, 7) float32
      obs['ee_states']:    (To, 10) float32
      action:              (T, 10) float32, where T == horizon
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.1,
        is_val: bool = False,
        rgb_keys=("images_0",),
        lowdim_keys=("joint_states", "ee_states"),
        max_train_episodes: Optional[int] = None,
    ):
        self.zarr_path = zarr_path
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.is_val = bool(is_val)

        self.rgb_keys = list(rgb_keys)
        if len(self.rgb_keys) == 0:
            raise ValueError("rgb_keys must contain at least one camera key, e.g. ['images_0'].")
        self.max_train_episodes = max_train_episodes
        self.lowdim_keys = list(lowdim_keys)
        if len(self.lowdim_keys) == 0:
            raise ValueError("lowdim_keys must contain at least one key, e.g. ['joint_states'].")

        root = zarr.open(zarr_path, mode="r")

        # --- arrays ---
        self.action = root["data"]["action"]
        self.episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

        # Load all lowdim arrays like data/lowdim_joint_states, data/lowdim_ee_states, ...
        self.lowdim_arrays = {}
        for k in self.lowdim_keys:
            zkey = f"lowdim_{k}"
            if zkey not in root["data"]:
                raise KeyError(
                    f"Missing Zarr dataset data/{zkey}. "
                    f"Expected because lowdim_keys includes '{k}'."
                )
            self.lowdim_arrays[k] = root["data"][zkey]

        # Load all camera arrays like data/rgb_images_0, data/rgb_images_1, ...
        self.rgb_arrays = {}
        for k in self.rgb_keys:
            zkey = f"rgb_{k}"
            if zkey not in root["data"]:
                raise KeyError(
                    f"Missing Zarr dataset data/{zkey}. "
                    f"Expected because rgb_keys includes '{k}'."
                )
            self.rgb_arrays[k] = root["data"][zkey]

        # safety checks
        if self.action.shape[1] != 10:
            raise ValueError(f"Expected action dim=10, got {self.action.shape[1]}")
        if self.horizon < self.n_obs_steps + self.n_action_steps - 1:
            raise ValueError(
                "horizon must be >= n_obs_steps + n_action_steps - 1 "
                f"(got horizon={self.horizon}, n_obs_steps={self.n_obs_steps}, "
                f"n_action_steps={self.n_action_steps})"
            )

        T = self.action.shape[0]
        for arr in self.lowdim_arrays.values():
            if arr.shape[0] != T:
                raise ValueError("Time length mismatch: lowdim and action")
        for arr in self.rgb_arrays.values():
            if arr.shape[0] != T:
                raise ValueError("Time length mismatch: rgb and action")
            if arr.shape[-1] != 3:
                raise ValueError("Expected rgb last dim=3 (H,W,3)")

        # split episodes into train/val by episode, not by timestep (no leakage)
        n_eps = len(self.episode_ends)
        val_mask = get_val_mask(n_eps, val_ratio=self.val_ratio, seed=self.seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_n=self.max_train_episodes, seed=self.seed)
        episode_mask = val_mask if self.is_val else train_mask

        if np.any(episode_mask):
            self.indices = create_indices(
                episode_ends=self.episode_ends,
                sequence_length=self.horizon,
                episode_mask=episode_mask,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                debug=True,
            )
        else:
            self.indices = np.zeros((0, 4), dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def _sample_sequence(self, arr, buffer_start, buffer_end, sample_start, sample_end):
        sample = np.asarray(arr[buffer_start:buffer_end])
        if (sample_start > 0) or (sample_end < self.horizon):
            data = np.zeros((self.horizon,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start > 0:
                data[:sample_start] = sample[0]
            if sample_end < self.horizon:
                data[sample_end:] = sample[-1]
            data[sample_start:sample_end] = sample
            return data
        return sample

    def __getitem__(self, idx):
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]

        act = self._sample_sequence(self.action, buffer_start, buffer_end, sample_start, sample_end)

        obs = {}
        for key, arr in self.rgb_arrays.items():
            rgb = self._sample_sequence(arr, buffer_start, buffer_end, sample_start, sample_end)
            rgb = rgb[:self.n_obs_steps]
            rgb = rgb.astype(np.float32) / 255.0
            rgb = np.moveaxis(rgb, -1, 1)  # T,C,H,W
            obs[key] = torch.from_numpy(rgb)

        for key, arr in self.lowdim_arrays.items():
            lowdim = self._sample_sequence(arr, buffer_start, buffer_end, sample_start, sample_end)
            lowdim = lowdim[:self.n_obs_steps].astype(np.float32)
            obs[key] = torch.from_numpy(lowdim)

        act = act.astype(np.float32)

        return {
            "obs": obs,
            "action": torch.from_numpy(act),  # (T, 10)
        }

    def get_validation_dataset(self):
        return SpotZarrImageDataset(
            zarr_path=self.zarr_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=self.seed,
            val_ratio=self.val_ratio,
            is_val=True,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            max_train_episodes=None,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.action)
        for key, arr in self.lowdim_arrays.items():
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(arr)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.action))
