"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
from abc import ABC, abstractmethod

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
        columns_winner = [
            'date',
            'winner_tie',
            'h_a',
            'pts_allowed_winner',
            'yds_gained',
            'tos'
        ]

        columns_loser = [
            'date',
            'loserttie',
            'h_a',
            'pts_allowed_loser',
            'yds_allowed',
            'opp_tos',
        ]

        data_avg_winner = data[columns_winner]
        data_avg_loser = data[columns_loser]

        data_avg_winner.rename(columns={
            'winner_tie': 'team',
            'h_a': 'home',
            'pts_allowed_winner': 'pts',
            'yds_gained': 'yds',
            'tos' : 'turnovers'
        }, inplace=True)

        data_avg_loser.rename(columns={
            'loserttie': 'team',
            'h_a': 'home',
            'pts_allowed_loser': 'pts',
            'yds_allowed': 'yds',
            'opp_tos' : 'turnovers'
        }, inplace=True)

        data = pd.concat([data_avg_loser, data_avg_winner])
        data['home'] = data['home'].map({'@' : 1, '' : 0})

        num_columns = ['pts', 'yds', 'turnovers']
        for column in num_columns:
            data[column] = data[column].astype('int')

        grouped_data = data.groupby(['team', 'home'], as_index=False).mean(numeric_only=True)        
        return grouped_data

ModellingKind = ModelTeamAvg