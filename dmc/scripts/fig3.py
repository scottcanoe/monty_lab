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
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
    load_object_model,
)
from plot_utils import (
    TBP_COLORS,
    SensorModuleData,
    axes3d_clean,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
    violinplot,
)
from scipy.spatial.transform import Rotation as R

init_matplotlib_style()


# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

"""
--------------------------------------------------------------------------------
Panel A: Sensor path and known objects
--------------------------------------------------------------------------------
"""


def plot_sensor_path():
    """Plot the sensor path for panel A."""
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={"projection": "3d"})

    model = load_object_model("dist_agent_1lm", "mug")
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=model.rgba,
        alpha=0.35,
        s=4,
        edgecolor="none",
    )
    sm = SensorModuleData(stats["SM_0"])
    sm.plot_sensor_path(ax, steps=36)

    ax.set_proj_type("persp", focal_length=0.5)
    ax.view_init(115, -90, 0)
    axes3d_clean(ax)
    axes3d_set_aspect_equal(ax)
    plt.show()
    out_dir = OUT_DIR / "sensor_path"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sensor_path.png", dpi=300)
    fig.savefig(out_dir / "sensor_path.svg")


def plot_known_objects():
    """Plot the "known objects" for panel A."""
    fig, axes = plt.subplots(1, 3, figsize=(5, 4), subplot_kw={"projection": "3d"})
    mug = load_object_model("dist_agent_1lm", "mug")
    bowl = load_object_model("dist_agent_1lm", "bowl")
    golf_ball = load_object_model("dist_agent_1lm", "golf_ball")

    axes[0].scatter(mug.x, mug.y, mug.z, color=mug.rgba, alpha=0.5, s=5, linewidth=0)
    axes[1].scatter(
        bowl.x, bowl.y, bowl.z, color=bowl.rgba, alpha=0.5, s=5, linewidth=0
    )
    axes[2].scatter(
        golf_ball.x,
        golf_ball.y,
        golf_ball.z,
        color=golf_ball.rgba,
        alpha=0.5,
        s=5,
        linewidth=0,
    )

    for ax in axes:
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.view_init(115, -90, 0)

    fig.savefig(OUT_DIR / "known_objects.png", dpi=300)
    fig.savefig(OUT_DIR / "known_objects.svg")
    plt.show()


"""
--------------------------------------------------------------------------------
Panel B: Evidence graphs
--------------------------------------------------------------------------------
"""


def plot_evidence_graphs_and_patches():
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig3_evidence_run"
    detailed_stats = DetailedJSONStatsInterface(
        experiment_dir / "detailed_run_stats.json"
    )
    stats = detailed_stats[0]

    object_names = ["mug", "bowl", "golf_ball"]
    steps = np.array([0, 10, 20, 39, 40])
    steps = np.arange(41)
    n_steps = len(steps)

    n_rows = n_cols = 64
    center_loc = n_rows // 2 * n_cols + n_cols // 2
    centers = np.zeros((n_steps, 3))
    for i in range(n_steps):
        arr = np.array(stats["SM_0"]["raw_observations"][i]["semantic_3d"])
        centers[i, 0] = arr[center_loc, 0]
        centers[i, 1] = arr[center_loc, 1]
        centers[i, 2] = arr[center_loc, 2]

    # Extract evidence values for all objects.
    all_evidences = stats["LM_0"]["evidences"]
    all_possible_locations = stats["LM_0"]["possible_locations"]
    all_possible_rotations = stats["LM_0"]["possible_rotations"]
    objects = {
        name: load_object_model("dist_agent_1lm_10distinctobj", name)
        for name in object_names
    }
    for name, obj in objects.items():
        obj.evidences, obj.locations = [], []
        for i in range(n_steps):
            obj.evidences.append(all_evidences[i][name])
            obj.locations.append(all_possible_locations[i][name])
        obj.evidences = np.array(obj.evidences)
        obj.locations = np.array(obj.locations)
        obj.rotation = np.array(all_possible_rotations[0][name])

    # Plot evidence graphs for each object and step individually.
    out_dir = OUT_DIR / "evidence_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(steps):
        fig, axes = plt.subplots(
            1, len(objects) + 1, figsize=(12, 6), subplot_kw=dict(projection="3d")
        )

        # Plot mug with current observation location.
        ax = axes[0]
        mug = objects["mug"]
        ax.scatter(
            mug.x,
            mug.y,
            mug.z,
            color="gray",
            alpha=0.1,
            s=1,
            linewidths=0,
        )
        center = centers[step]
        ax.scatter(center[0], center[1], center[2], color="red", s=50)
        ax.view_init(100, -100, -10)
        axes3d_clean(ax, grid=False)
        axes3d_set_aspect_equal(ax)
        ax.set_title(f"step {step}")
        ax.set_xlim(-0.119, 0.119)
        ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
        ax.set_zlim(-0.119, 0.119)

        # Make colormap for this step.
        all_evidences = [obj.evidences[step].flatten() for obj in objects.values()]
        all_evidences = np.concatenate(all_evidences)
        evidences_min = np.percentile(all_evidences, 2.5)
        evidences_max = np.percentile(all_evidences, 99.99)
        scalar_map = plt.cm.ScalarMappable(
            cmap="inferno", norm=plt.Normalize(vmin=evidences_min, vmax=evidences_max)
        )

        for j, obj in enumerate(objects.values()):
            ax = axes[j + 1]
            locations = obj.locations[step]
            n_points = locations.shape[0] // 2
            locations = locations[:n_points]
            evidences = obj.evidences[step]
            ev1 = evidences[:n_points]
            ev2 = evidences[n_points:]
            stacked = np.hstack([ev1[:, np.newaxis], ev2[:, np.newaxis]])
            evidences = stacked.max(axis=1)

            colors = evidences
            sizes = np.log(evidences - evidences.min() + 1) * 10
            alphas = np.array(evidences)
            alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
            x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
            ax.scatter(
                x,
                y,
                z,
                c=colors,
                cmap="inferno",
                alpha=alphas,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0,
            )

            # Plot highest evidence location separately.
            ind_ev_max = evidences.argmax()
            ev = evidences[ind_ev_max]
            x, y, z = x[ind_ev_max], y[ind_ev_max], z[ind_ev_max]
            sizes = sizes[ind_ev_max]
            ax.scatter(
                x,
                y,
                z,
                c=ev,
                cmap="inferno",
                alpha=1,
                vmin=evidences_min,
                vmax=evidences_max,
                s=sizes,
                linewidths=0,
            )

            ax.view_init(100, -100, -10)
            axes3d_clean(ax, grid=False)
            axes3d_set_aspect_equal(ax)
            ax.set_title(f"step {step}")
            ax.set_xlim(-0.119, 0.119)
            ax.set_ylim(1.5 - 0.119, 1.5 + 0.119)
            ax.set_zlim(-0.119, 0.119)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
        fig.savefig(png_dir / f"evidence_graphs_{step}.png", dpi=300)
        fig.savefig(svg_dir / f"evidence_graphs_{step}.svg")
        plt.show()
        plt.close(fig)

    # Plot the colorbar.
    fig, ax = plt.subplots(1, 1, figsize=(1, 2))
    cbar = plt.colorbar(scalar_map, ax=ax, orientation="vertical", label="Evidence")
    ax.remove()  # Remove the empty axes, we just want the colorbar
    cbar.set_ticks([])
    cbar.set_label("")
    fig.tight_layout()
    fig.savefig(out_dir / "colorbar.png", dpi=300)
    fig.savefig(out_dir / "colorbar.svg")

    # Extract RGBA patches for sensor module 0.
    rgba_patches = []
    for ind in range(n_steps):
        rgba_patches.append(np.array(stats["SM_0"]["raw_observations"][ind]["rgba"]))
    rgba_patches = np.array(rgba_patches)

    # Save the RGBA patches.
    out_dir = OUT_DIR / "patches"
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        patch = rgba_patches[step]
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(patch)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        fig.savefig(png_dir / f"patch_step_{step}.png", dpi=300)
        fig.savefig(svg_dir / f"patch_step_{step}.svg")
        plt.close(fig)


"""
--------------------------------------------------------------------------------
Panel C: ?
--------------------------------------------------------------------------------
"""


def plot_performance() -> None:
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_1lm_noise"),
        load_eval_stats("dist_agent_1lm_randrot_all"),
        load_eval_stats("dist_agent_1lm_randrot_all_noise"),
    ]
    percent_correct = np.zeros(len(dataframes))
    rotation_errors = np.zeros(len(dataframes), dtype=object)
    for i, df in enumerate(dataframes):
        sub_df = df[df.primary_performance.isin(["correct", "correct_mlh"])]
        percent_correct[i] = 100 * len(sub_df) / len(df)
        rotation_errors[i] = np.degrees(sub_df.rotation_error)

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
        body.set_facecolor(TBP_COLORS["purple"])
        body.set_alpha(1.0)
    vp["cmedians"].set_color("black")
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Rotation Error (deg)")

    ax1.set_xticks([0.5, 2.5, 4.5, 6.5])
    xticklabels = ["base", "noise", "RR", "noise + RR"]
    ax1.set_xticklabels(xticklabels, rotation=0, ha="center")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=300)
    fig.savefig(out_dir / "performance.svg")
    plt.show()


def draw_randrot_noise_icons():
    model = load_object_model("dist_agent_1lm", "mug")
    model = model - [0, 1.5, 0]
    fig, axes = plt.subplots(1, 4, figsize=(8, 4), subplot_kw={"projection": "3d"})

    # Params
    params = {
        "alpha": 0.45,
        "edgecolors": "none",
        "s": 2,
    }
    loc_std = 0.002  # same parameter as in configs

    # Use bluementa as base color.
    hex = TBP_COLORS["blue"].lstrip("#")
    rgba = np.array([int(hex[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0])
    rgba = np.tile(rgba, (len(model.x), 1))

    # Make noisy color.
    rgba_noise = rgba.copy()
    rgba_noise = rgba_noise + 0.3 * np.random.randn(len(rgba_noise), 4)
    rgba_noise = np.clip(rgba_noise, 0, 1)
    rgba_noise[:, 3] = 1
    loc_noise = loc_std * np.random.randn(*model.pos.shape)

    # - draw base model
    ax = axes[0]
    ax.scatter(model.x, model.y, model.z, color=rgba, **params)

    # - draw noise
    ax = axes[1]
    noise_model = model.copy()
    noise_model.pos += loc_noise
    ax.scatter(
        noise_model.x,
        noise_model.y,
        noise_model.z,
        color=rgba_noise,
        **params,
    )

    # - draw random rotation
    ax = axes[2]
    randrot_model = model.rotated([45, 10, 30], degrees=True)
    ax.scatter(
        randrot_model.x,
        randrot_model.y,
        randrot_model.z,
        color=rgba,
        **params,
    )

    # - draw radom rotation + noise
    ax = axes[3]
    randrot_noise_model = model.rotated([25, 30, -135], degrees=True)
    randrot_noise_model.pos += loc_noise
    ax.scatter(
        randrot_noise_model.x,
        randrot_noise_model.y,
        randrot_noise_model.z,
        color=rgba_noise,
        **params,
    )

    # Clean up
    for ax in axes:
        axes3d_set_aspect_equal(ax)
        ax.axis("off")
        ax.view_init(90, -90)
        ax.axis("off")

    plt.show()
    fig.savefig(OUT_DIR / "randrot_noise_icons.png")
    fig.savefig(OUT_DIR / "randrot_noise_icons.svg")


# plot_performance()
# plot_evidence_graphs_and_patches()
# plot_known_objects()
# plot_sensor_path()
# draw_randrot_noise_icons()

out_dir = OUT_DIR / "performance"
out_dir.mkdir(parents=True, exist_ok=True)
dataframes = [
    load_eval_stats("dist_agent_1lm"),
    load_eval_stats("dist_agent_1lm_noise"),
    load_eval_stats("dist_agent_1lm_randrot_all"),
    load_eval_stats("dist_agent_1lm_randrot_all_noise"),
]
accuracy, rotation_error = [], []
for i, df in enumerate(dataframes):
    sub_df = df[df.primary_performance.isin(["correct", "correct_mlh"])]
    accuracy.append(100 * len(sub_df) / len(df))
    rotation_error.append(np.degrees(sub_df.rotation_error))

fig, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
ax2 = ax1.twinx()

bar_width = 0.4
violin_width = 0.4
gap = 0.02
xticks = np.arange(4) * 1.3
bar_positions = xticks - bar_width / 2 - gap / 2
violin_positions = xticks + violin_width / 2 + gap / 2
median_style = dict(color="lightgray", lw=1, ls="-")

# Plot accuracy bars
ax1.bar(
    bar_positions,
    accuracy,
    color=TBP_COLORS["blue"],
    width=bar_width,
)
ax1.set_ylim(0, 100)
ax1.set_ylabel("% Correct")

# Plot rotation error violins
violinplot(
    rotation_error,
    violin_positions,
    width=violin_width,
    color=TBP_COLORS["purple"],
    showextrema=False,
    showmedians=True,
    median_style=median_style,
    ax=ax2,
)


ax2.set_yticks([0, 45, 90, 135, 180])
ax2.set_ylim(0, 180)
ax2.set_ylabel("Rotation Error (deg)")

ax1.set_xticks(xticks)
xticklabels = ["base", "noise", "RR", "noise + RR"]
ax1.set_xticklabels(xticklabels, rotation=0, ha="center")

ax1.spines["right"].set_visible(True)
ax2.spines["right"].set_visible(True)

fig.tight_layout()
fig.savefig(out_dir / "performance.png", dpi=300)
fig.savefig(out_dir / "performance.svg")
plt.show()
