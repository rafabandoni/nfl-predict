"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from nfl_predict.io import datasets, services
from nfl_predict.jobs import base

# %% JOBS


class DataEngineerJob(base.Job):
    """Todo
    """

    KIND: T.Literal["DataEngineerJob"] = "DataEngineerJob"

    # Inputs
    input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        logger.info("Reading: {}", self.input.KIND)
        data = self.input.read()
        logger.debug("Dataframe shape: {}", data.shape)
        self.output.write(data)
        logger.info("File saved in {}", self.output.path)