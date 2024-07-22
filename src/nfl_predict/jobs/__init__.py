"""High-level jobs of the project."""

# %% IMPORTS

from nfl_predict.jobs.dataengineer import DataEngineerJob
from nfl_predict.jobs.modelling import Modelling

# %% TYPES

JobKind = DataEngineerJob | Modelling

# %% EXPORTS

__all__ = ["DataEngineerJob", "ModellingJob", "JobKind"]
