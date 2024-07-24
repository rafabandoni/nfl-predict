"""High-level jobs of the project."""

# %% IMPORTS

from nfl_predict.jobs.dataengineer import DataEngineerJob
from nfl_predict.jobs.modelling import Modelling
from nfl_predict.jobs.training import Training

# %% TYPES

JobKind = DataEngineerJob | Modelling | Training

# %% EXPORTS

__all__ = ["DataEngineerJob", "ModellingJob", "TrainingJob", "JobKind"]
