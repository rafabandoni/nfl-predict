"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import pydantic as pdt

from nfl_predict.io import datasets
from nfl_predict.core import predicting
from nfl_predict.jobs import base

# %% JOBS


class Predicting(base.Job):
    """Todo
    """

    KIND: T.Literal["PredictingJob"] = "PredictingJob"

    # Inputs
    to_predict_input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    averages_input: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    predict_kind: predicting.PredictingKind = pdt.Field(..., discriminator="KIND")
    output: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    @T.override
    def run(self) -> base.Locals:
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)

        logger.info("Reading: {}", self.to_predict_input.path)
        data_to_predict = self.to_predict_input.read()
        logger.debug("Dataframe shape: {}", data_to_predict.shape)

        logger.info("Loading averages data.")
        averages_data = self.averages_input.read()
        logger.debug("Dataframe shape: {}", averages_data.shape)

        logger.info("Predicting data using {}", self.predict_kind.model)
        predicted_data = self.predict_kind.predict(data_to_predict, averages_data)
        logger.debug("Dataframe shape: {}", predicted_data.shape)

        self.output.write(predicted_data)
        logger.info("File saved in {}", self.output.path)