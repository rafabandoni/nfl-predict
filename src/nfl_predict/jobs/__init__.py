"""High-level jobs of the project."""

# %% IMPORTS

from nfl_predict.jobs.dataengineer import DataEngineerJob
from nfl_predict.jobs.modelling import Modelling
from nfl_predict.jobs.training import Training
from nfl_predict.jobs.predicting import Predicting

# %% TYPES

JobKind = DataEngineerJob | Modelling | Training | Predicting

# %% EXPORTS

__all__ = ["DataEngineerJob", "ModellingJob", "TrainingJob", "PredictingJob", "JobKind"]
