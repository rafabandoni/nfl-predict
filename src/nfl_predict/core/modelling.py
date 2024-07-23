"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
from abc import ABC, abstractmethod

from nfl_predict.core.utils.modelling_func import ModellingFunctions

# %% READERS

class Modelling(ABC, pdt.BaseModel, strict=True, frozen=True, extra='forbid'):
    """
    Base class to read data from dataset.

    Used to get data from the dataset and load as dataframe into memory.

    Parameters:
        limit (int, optional): maximum number of rows to read. Defautls to none.
    """
    
    KIND: str

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """"
        Reads a dataframe from a dataset.

        Returns:
            pd.Dataframe: dataframe representation.
        """


class ModelTeamAvg(Modelling, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["ModelTeamAvg"] = "ModelTeamAvg"

    @T.override
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        modelling = ModellingFunctions()
        df_pts_per_year = modelling.create_pts_per_year(data)
        df = modelling.join_pts_per_year(data, df_pts_per_year)
        df = modelling.fix_columns(df)
        df = modelling.fix_home_column(df)
        outcome_df = modelling.create_outcomes_df(df)
        return outcome_df

ModellingKind = ModelTeamAvg