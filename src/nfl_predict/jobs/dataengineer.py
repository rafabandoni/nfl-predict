"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from nfl_predict.io import datasets, registries, services
from nfl_predict.jobs import base

# %% JOBS


class DataEngineerJob(base.Job):
    """Todo
    """

    KIND: T.Literal["DataEngineerJob"] = "DataEngineerJob"

    # Inputs
    # TODO: input de scrapping e de leitura de parquet
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")


    output: datasets.DatasetKind = pdt.Field(..., discriminator="KIND") # TODO

    @T.override
    def run(self) -> base.Locals:
        pass # TODO