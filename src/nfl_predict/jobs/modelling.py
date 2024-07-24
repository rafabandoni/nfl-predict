"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from nfl_predict.io import datasets, services
from nfl_predict.core import modelling
from nfl_predict.jobs import base

# %% JOBS


class Modelling(base.Job):
    """Todo
    """

    KIND: T.Literal["ModellingJob"] = "ModellingJob"

    # Inputs
    input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    model: modelling.ModellingKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        logger.info("Reading: {}", self.input.path)
        data = self.input.read()
        logger.debug("Dataframe shape: {}", data.shape)

        logger.info("Running model: {}", self.model.KIND)
        to_output = self.model.run(data)
        logger.debug("Output shape: {}", to_output.shape)
        
        self.output.write(to_output)
        logger.info("File saved in {}", self.output.path)