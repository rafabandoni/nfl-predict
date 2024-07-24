"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from nfl_predict.io import datasets, services
from nfl_predict.core import training
from nfl_predict.jobs import base

# %% JOBS


class Training(base.Job):
    """Todo
    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Inputs
    input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    model: training.TrainingKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        logger.info("Reading: {}", self.input.path)
        data = self.input.read()
        logger.debug("Dataframe shape: {}", data.shape)
        model = self.model.run(data)
        
        self.output.write(model)
        logger.info("File saved in {}", self.output.path)