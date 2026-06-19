# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "packages" / "cosmos-cuda"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch

from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
)
from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import EETrajectoryHead


class _Config:
    ee_head = {
        "enabled": True,
        "loss_weight": 0.05,
        "position_loss_weight": 1.0,
        "rotation_6d_loss_weight": 1.0,
        "gripper_loss_weight": 1.0,
    }


def main():
    batch_size = 2
    latent_frames = 5
    num_future_frames = 16
    hidden_dim = 32
    feature_dim = 24

    head = EETrajectoryHead(
        in_features=feature_dim,
        latent_frames=latent_frames,
        num_future_frames=num_future_frames,
        hidden_features=hidden_dim,
    )
    features = torch.randn(batch_size, latent_frames, 3, 4, feature_dim)
    pred = head(features)
    assert tuple(pred["position"].shape) == (batch_size, num_future_frames, 2, 3)
    assert tuple(pred["rotation_6d"].shape) == (batch_size, num_future_frames, 2, 6)
    assert tuple(pred["gripper_logits"].shape) == (batch_size, num_future_frames, 2)

    model = ActionVideo2WorldModelRectifiedFlow.__new__(ActionVideo2WorldModelRectifiedFlow)
    model.config = _Config()
    data_batch = {
        "ee_target_position": torch.randn(batch_size, num_future_frames, 2, 3),
        "ee_target_rotation_6d": torch.randn(batch_size, num_future_frames, 2, 6),
        "ee_target_gripper": torch.randint(0, 2, (batch_size, num_future_frames, 2)).float(),
    }
    loss, metrics = ActionVideo2WorldModelRectifiedFlow._ee_loss(model, pred, data_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for key in ("ee_position_loss", "ee_rotation_6d_loss", "ee_gripper_loss", "ee_loss_raw", "ee_loss"):
        assert key in metrics
        assert metrics[key].ndim == 0
        assert torch.isfinite(metrics[key])

    assert ActionVideo2WorldModelRectifiedFlow._ee_head_enabled(model) is True
    model.config.ee_head = {"enabled": False}
    assert ActionVideo2WorldModelRectifiedFlow._ee_head_enabled(model) is False
    print("[INFO] EE head smoke OK")


if __name__ == "__main__":
    main()
