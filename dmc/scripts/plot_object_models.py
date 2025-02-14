# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Get overview plots for DMC experiments.

This script generates basic figures for each set of experiments displaying number
of monty matching steps, accuracy, and rotation error. If functions are called with
`save=True`, figures and tables are saved under `DMC_ANALYSIS_DIR / overview`.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_PRETRAIN_DIR,
)
from plot_utils import axes3d_clean, axes3d_set_aspect_equal

plt.rcParams["font.size"] = 8

OUT_DIR = DMC_ANALYSIS_DIR / "object_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_77_object_models(model_name: str):
    inches_per_subplot = 2.5
    n_rows, n_cols = 11, 7
    h_pad = 0.25
    v_pad = 0.25
    width = n_cols * inches_per_subplot + (n_cols + 1) * h_pad
    height = n_rows * inches_per_subplot + (n_rows + 1) * v_pad

    model_path = DMC_PRETRAIN_DIR / model_name / "pretrained/model.pt"
    data = torch.load(model_path)
    data = data["lm_dict"][0]["graph_memory"]
    object_names = list(sorted(data.keys()))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), subplot_kw={"projection": "3d"}
    )
    for i, ax in enumerate(axes.flatten()):
        object_name = object_names[i]
        object_data = data[object_name]
        patch = object_data["patch"]
        points = np.array(patch.pos)
        if "rgba" in patch.feature_mapping:
            rgba_idx = patch.feature_mapping["rgba"]
            rgba = np.array(patch.x[:, rgba_idx[0] : rgba_idx[1]]) / 255.0
        else:
            rgba = None
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=rgba, alpha=1)
        axes3d_clean(ax)
        axes3d_set_aspect_equal(ax)
        n_points = len(points)
        ax.set_title(f"{object_name} ({n_points} points)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name}.png", dpi=300)
    plt.close(fig)


def plot_object_models_for_checkpoints():
    savedir = OUT_DIR / "checkpoints"
    savedir.mkdir(parents=True, exist_ok=True)

    model_name = "dist_agent_1lm_checkpoints"
    checkpoint = 1
    model_path = (
        DMC_PRETRAIN_DIR / model_name / f"pretrained/checkpoints/{checkpoint}/model.pt"
    )
    data = torch.load(model_path)
    object_names = list(sorted(data["lm_dict"][0]["graph_memory"].keys()))

    inches_per_subplot = 2.5
    n_rows, n_cols = 8, 4
    h_pad = 0.25
    v_pad = 0.25
    width = n_cols * inches_per_subplot + (n_cols + 1) * h_pad
    height = n_rows * inches_per_subplot + (n_rows + 1) * v_pad

    for object_name in object_names:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(width, height), subplot_kw={"projection": "3d"}
        )
        for i, ax in enumerate(axes.flatten()):
            checkpoint = i + 1
            model_path = (
                DMC_PRETRAIN_DIR
                / model_name
                / f"pretrained/checkpoints/{checkpoint}/model.pt"
            )
            data = torch.load(model_path)
            object_data = data["lm_dict"][0]["graph_memory"][object_name]
            patch = object_data["patch"]
            points = np.array(patch.pos)
            if "rgba" in patch.feature_mapping:
                rgba_idx = patch.feature_mapping["rgba"]
                rgba = np.array(patch.x[:, rgba_idx[0] : rgba_idx[1]]) / 255.0
            else:
                rgba = None
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=rgba, alpha=1)
            axes3d_clean(ax)
            axes3d_set_aspect_equal(ax)
            n_points = len(points)
            ax.set_title(f"{checkpoint} ({n_points} points)")
        fig.tight_layout()
        fig.savefig(savedir / f"{object_name}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    plot_77_object_models("dist_agent_1lm")
    plot_77_object_models("surf_agent_1lm")
    plot_77_object_models("touch_agent_1lm")
    plot_object_models_for_checkpoints()
