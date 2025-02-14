# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Configs for visualizations (not core experiments)

This file contains configs defined solely for making visualizations that go into
paper figures. The configs defined are:

- `visualize_8lm_patches`: An experiment that used to save patch view of an object
for the 8-patch distant agent model. Only runs one episode (using the mug). The
output is read and plotted by `scripts/visualize_multilm_patches.py`.

"""
import os
from copy import deepcopy

from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler

from .common import (
    DMC_PRETRAIN_DIR,
    DMC_RESULTS_DIR,
    DMC_ROOT_DIR,
    DMCEvalLoggingConfig,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig4_rapid_inference_with_voting import dist_agent_8lm_half_lms_match
from .fig9_structured_object_representations import (
    EvidenceLoggingMontyObjectRecognitionExperiment,
)

# Main output directory for visualization experiment results.
VISUALIZATIONS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")


fig3_evidence_run = deepcopy(dist_agent_1lm)
fig3_evidence_run.update(
    dict(
        experiment_class=EvidenceLoggingMontyObjectRecognitionExperiment,
        experiment_args=EvalExperimentArgs(
            model_name_or_path=str(
                DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
            ),
            n_eval_epochs=1,
            max_total_steps=100,
            max_eval_steps=100,
        ),
        logging_config=DetailedEvidenceLMLoggingConfig(
            output_dir=str(VISUALIZATIONS_DIR),
            run_name="fig3_evidence_run",
            wandb_group="dmc",
            monty_log_level="SELECTIVE",
        ),
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
        ),
    )
)
fig3_evidence_run["monty_config"].monty_args.min_eval_steps = 41
fig3_evidence_run[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False


fig4_visualize_8lm_patches = deepcopy(dist_agent_8lm_half_lms_match)
fig4_visualize_8lm_patches["logging_config"].run_name = "fig4_visualize_8lm_patches"
fig4_visualize_8lm_patches["logging_config"].output_dir = VISUALIZATIONS_DIR
fig4_visualize_8lm_patches["logging_config"].monty_handlers.append(DetailedJSONHandler)
fig4_visualize_8lm_patches["experiment_args"].n_eval_epochs = 1
fig4_visualize_8lm_patches["experiment_args"].max_total_steps = 1
fig4_visualize_8lm_patches["experiment_args"].max_eval_steps = 1
fig4_visualize_8lm_patches["monty_config"].monty_args.num_exploration_steps = 1
fig4_visualize_8lm_patches["eval_dataloader_args"].object_names = ["mug"]
fig4_visualize_8lm_patches["eval_dataloader_args"].object_init_sampler.rotations = [
    [0, 0, 0]
]
# Set viewfinder resolution to 256 x 256 for a denser "background" image. The remaining
# patches uses the same resolution as the original 64 x 64 patches.
resolutions = [[64, 64]] * 9
resolutions[-1] = [256, 256]
dataset_args = fig4_visualize_8lm_patches["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = resolutions
dataset_args.__post_init__()

CONFIGS = {
    "fig3_evidence_run": fig3_evidence_run,
    "fig4_visualize_8lm_patches": fig4_visualize_8lm_patches,
}
