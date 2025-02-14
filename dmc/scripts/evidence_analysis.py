import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from data_utils import (
    DMC_RESULTS_DIR,
    DetailedJSONStatsInterface,
)

experiment_dir = DMC_RESULTS_DIR / "dist_agent_1lm_randrot_noise_10simobj"
stats = DetailedJSONStatsInterface(experiment_dir / "detailed_run_stats.json")
ep = stats[0]
