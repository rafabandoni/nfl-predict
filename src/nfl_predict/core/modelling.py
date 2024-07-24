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


class ModelTrainData(Modelling, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["ModelTrainData"] = "ModelTrainData"

    @T.override
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        modelling = ModellingFunctions()
        df_pts_per_year = modelling.create_pts_per_year(data)
        df = modelling.join_pts_per_year(data, df_pts_per_year)
        df = modelling.fix_columns(df)
        df = modelling.fix_home_column(df)
        outcome_df = modelling.create_outcomes_df(df)
        return outcome_df
    
class ModelTeamAvg(Modelling, strict=True, frozen=True, extra="forbid"):
    """
    
    """

    KIND: T.Literal["ModelTeamAvg"]
    
    @T.override
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data['year'] = data['date'].str[:4]
        data_winner = data[['year', 'winner_tie', 'h_a', 'pts_allowed_winner', 'yds_gained', 'tos']]
        data_loser = data[['year', 'loserttie', 'h_a', 'pts_allowed_loser', 'yds_allowed', 'opp_tos']]

        new_columns = ['year', 'team', 'home', 'pts', 'yds', 'turnovers']
        data_winner.columns = new_columns
        data_loser.columns = new_columns

        data_concat = pd.concat([data_winner, data_loser])
        data_concat['home'] = data_concat['home'] == '@'

        for column in ['pts', 'yds', 'turnovers']:
            data_concat[column] = data_concat[column].astype('int')

        df_grouped = data_concat.groupby(['year', 'team', 'home'], as_index=False).mean().round(0)
        return df_grouped 

ModellingKind = ModelTrainData | ModelTeamAvg