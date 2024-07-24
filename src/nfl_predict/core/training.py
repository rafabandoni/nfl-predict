"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
from abc import ABC, abstractmethod

from sklearn import tree

# %% READERS

class Training(ABC, pdt.BaseModel, strict=True, frozen=True, extra='forbid'):
    """
    Base class to read data from dataset.

    Used to get data from the dataset and load as dataframe into memory.

    Parameters:
        limit (int, optional): maximum number of rows to read. Defautls to none.
    """
    
    KIND: str

    @abstractmethod
    def run(self, data: pd.DataFrame):
        """"
        Reads a dataframe from a dataset.

        Returns:
            pd.Dataframe: dataframe representation.
        """


class TrainingData(Training, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["TrainingData"] = "TrainingData"

    @T.override
    def run(self, data: pd.DataFrame):
        features = [
            'team_a_yds',
            'team_a_turnovers',
            'team_b_yds',
            'team_b_turnovers',
            'pts_per_year_team_a',
            'pts_per_year_team_b'
        ]

        X = data[features]
        y = data['outcome']

        arvore = tree.DecisionTreeClassifier()
        arvore.fit(X,y)
        return arvore

TrainingKind = TrainingData