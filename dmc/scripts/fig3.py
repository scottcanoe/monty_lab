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

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_PRETRAIN_DIR,
    DMC_RESULTS_DIR,
    DMC_ROOT_DIR,
    VISUALIZATIONS_DIR,
    get_percent_correct,
    load_eval_stats,
)
from evidence_analysis import DetailedJSONStatsInterface, describe_dict
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import ArrayLike
from plot_utils import TBP_COLORS, axes3d_clean, axes3d_set_aspect_equal
from tbp.monty.frameworks.models.object_model import GraphObjectModel

plt.rcParams["font.size"] = 8

# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_stats_plot():
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_1lm_noise"),
        load_eval_stats("dist_agent_1lm_randrot_all"),
        load_eval_stats("dist_agent_1lm_randrot_all_noise"),
    ]
    conditions = ["base", "noise", "RR", "noise + RR"]

    percent_correct = [get_percent_correct(df) for df in dataframes]
    rotation_errors = [np.rad2deg(df.rotation_error.dropna()) for df in dataframes]

    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2))

    # Plot accuracy bars
    ax1.bar(
        [0, 2, 4, 6],
        percent_correct,
        color=TBP_COLORS["blue"],
        width=0.8,
    )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% Correct")

    # Plot rotation error violins
    ax2 = ax1.twinx()
    vp = ax2.violinplot(
        rotation_errors,
        positions=[1, 3, 5, 7],
        showextrema=False,
        showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(TBP_COLORS["pink"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Error (deg)")

    ax1.set_xticks([0.5, 2.5, 4.5, 6.5])
    ax1.set_xticklabels(conditions, rotation=0, ha="center")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stats.png", dpi=300)
    fig.savefig(OUT_DIR / "stats.pdf")
    plt.show()


class ObjectModel:
    def __init__(
        self,
        points: ArrayLike,
        translation: ArrayLike = (0, 0, 0),
        rgba: Optional[Any] = None,
    ):
        self.points = np.asarray(points).astype(float)
        self.translation = np.asarray(translation).astype(float)
        self.rgba = rgba

    @property
    def x(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.points[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.points[:, 2]

    def centered(self) -> "ObjectModel":
        return self - self.translation

    def rotated(self, pitch: Number, roll: Number, yaw: Number) -> "ObjectModel":
        # Convert angles to radians
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        yaw = np.radians(yaw)

        # Create rotation matrices
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )
        # Combine rotations
        R = Rz @ Ry @ Rx

        # Undo translation, rotate, and reapply translation.
        centered = self.points - self.translation
        rotated = centered @ R.T
        points = rotated + self.translation

        # Assign points to the new object, and return it.
        out = deepcopy(self)
        out.points = points
        return out

    def __add__(self, translation: ArrayLike) -> "ObjectModel":
        translation = np.asarray(translation)
        out = deepcopy(self)
        out.points += translation
        out.translation += translation
        return out

    def __sub__(self, translation: ArrayLike) -> "ObjectModel":
        translation = np.asarray(translation)
        return self + (-translation)


def load_object_model(
    model_name: str,
    object_name: str,
    checkpoint: Optional[int] = None,
    lm_id: int = 0,
) -> ObjectModel:
    if checkpoint is None:
        model_path = DMC_PRETRAIN_DIR / model_name / "pretrained/model.pt"
    else:
        model_path = (
            DMC_PRETRAIN_DIR
            / model_name
            / f"pretrained/checkpoints/{checkpoint}/model.pt"
        )
    data = torch.load(model_path)
    data = data["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]
    points = np.array(data.pos)
    if "rgba" in data.feature_mapping:
        rgba_idx = data.feature_mapping["rgba"]
        rgba = np.array(data.x[:, rgba_idx[0] : rgba_idx[1]]) / 255.0
    else:
        rgba = None
    translation = np.array([0.0, 1.5, 0.0])
    out = ObjectModel(points, translation, rgba)
    return out


def to_mpl_coords(
    points: np.ndarray, translation: Optional[ArrayLike] = None
) -> np.ndarray:
    T = np.array(
        [
            [1, 0, 0],  # X stays the same
            [0, 0, -1],  # Y is now negative Z
            [0, -1, 0],  # Z is now negative Y
        ]
    )
    if translation is None:
        translation = np.array([0, 0, 0])
    else:
        translation = np.asarray(translation)
    centered_points = points - translation
    rotated_points = centered_points @ T.T
    return rotated_points + translation


def get_centered_mug():
    mug = load_object_model("dist_agent_1lm", "mug")

    # Move the mug's center to the origin. The center here is defined as the center
    # of the cylinder (more or less). The translation below is only appropriate for the
    # mug model.
    arg_zmin, arg_zmax = np.argmin(mug.z), np.argmax(mug.z)
    zmin, zmax = mug.z[arg_zmin], mug.z[arg_zmax]
    zcenter = np.mean([zmin, zmax])
    xmin, xmax = mug.x[arg_zmin], mug.x[arg_zmax]
    xcenter = np.mean([xmin, xmax])
    ymin = mug.y.min()
    offset = np.array([xcenter, ymin, zcenter])
    mug = mug - offset
    mug.translation = np.array([0, 0, 0])
    return mug


def mug_plot_top_left():
    mug = load_object_model("dist_agent_1lm", "mug")
    mug.translation = np.array([-0.012628763, 1.4593439, 0.00026388466])
    mug -= mug.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw mug at detected_rotation.
    mug_rot_detected = mug.rotated(0, 0, 0)
    ax.scatter(
        mug_rot_detected.x,
        mug_rot_detected.y,
        mug_rot_detected.z,
        color=blue,
        alpha=0.3,
    )

    # Draw mug at most_likely_rotation.
    mug_r = mug.rotated(0, 0, 180)
    ax.scatter(mug_r.x, mug_r.y, mug_r.z, color=green, alpha=0.3)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()

    out_path = OUT_DIR / "mug_top_left.png"
    fig.savefig(out_path, dpi=300)
    return fig


def mug_plot_top_right():
    mug = load_object_model("dist_agent_1lm", "mug")
    mug.translation = np.array([-0.012628763, 1.4593439, 0.00026388466])
    mug -= mug.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original mug.
    ax.scatter(mug.x, mug.y, mug.z, color=blue, alpha=0.3)

    # Draw rotated mug.
    mug_r = mug.rotated(15, 70, 45)
    ax.scatter(mug_r.x, mug_r.y, mug_r.z, color=green, alpha=0.3)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "mug_top_right.png"
    fig.savefig(out_path, dpi=300)

    return fig


def spoon_plot_bottom_left():
    spoon = load_object_model("surf_agent_1lm", "spoon")
    spoon -= spoon.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original spoon.
    spoon = spoon.rotated(0, 0, 20)
    ax.scatter(spoon.x, spoon.y, spoon.z, color=blue, alpha=0.8)

    # Draw rotated spoon.
    spoon_r = spoon.rotated(180, 0, 0)
    ax.scatter(spoon_r.x, spoon_r.y, spoon_r.z, color=green, alpha=0.8)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "spoon_bottom_left.png"
    fig.savefig(out_path, dpi=300)


def spoon_plot_bottom_right():
    spoon = load_object_model("surf_agent_1lm", "spoon")
    spoon -= spoon.translation

    blue = TBP_COLORS["blue"]
    green = TBP_COLORS["green"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Draw original spoon.
    ax.scatter(spoon.x, spoon.y, spoon.z, color=blue, alpha=0.75)

    # Draw rotated spoon.
    spoon_r = spoon.rotated(80, 70, 180)
    # spoon_r -= ()
    ax.scatter(spoon_r.x, spoon_r.y, spoon_r.z, color=green, alpha=0.75)

    ax.view_init(125, -100, -10)
    axes3d_clean(ax, label_axes=True)
    axes3d_set_aspect_equal(ax)

    fig.tight_layout()
    plt.show()
    out_path = OUT_DIR / "spoon_bottom_right.png"
    fig.savefig(out_path, dpi=300)


experiment_dir = VISUALIZATIONS_DIR / "fig3_evidence_run"
stats = DetailedJSONStatsInterface(experiment_dir / "detailed_run_stats.json")
ep = stats[0]

evidences = ep["LM_0"]["evidences_ls"]
possible_locations = ep["LM_0"]["possible_locations_ls"]
possible_rotations = ep["LM_0"]["possible_rotations_ls"]

object_name = "mug"
obj_evidences, obj_locations = [], []
for step in range(len(evidences)):
    obj_evidences.append(evidences[step][object_name])
    obj_locations.append(possible_locations[step][object_name])

obj_evidences = np.array(obj_evidences)
obj_locations = np.array(obj_locations)
obj_rotation = np.array(possible_rotations[0][object_name])
