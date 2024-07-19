"""High-level jobs of the project."""

# %% IMPORTS

from nfl_predict.jobs.dataengineer import DataEngineerJob

# %% TYPES

JobKind = DataEngineerJob

# %% EXPORTS

__all__ = ["DataEngineerJob", "JobKind"]
