"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
from abc import ABC, abstractmethod
import joblib

# %% READERS

class Predicting(ABC, pdt.BaseModel, strict=True, frozen=True, extra='forbid'):
    """
    Base class to read data from dataset.

    Used to get data from the dataset and load as dataframe into memory.

    Parameters:
        limit (int, optional): maximum number of rows to read. Defautls to none.
    """
    
    KIND: str

    model: str

    @abstractmethod
    def predict(self, data: pd.DataFrame, averages: pd.DataFrame) -> pd.DataFrame:
        """"
        Reads a dataframe from a dataset.

        Returns:
            pd.Dataframe: dataframe representation.
        """


class PredictingData(Predicting, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["PredictingData"] = "PredictingData"

    model: str

    @T.override
    def predict(self, data: pd.DataFrame, averages: pd.DataFrame) -> pd.DataFrame:
        num = ['pts', 'yds', 'turnovers']
        for i in num:
            averages[i] = averages[i].astype('int')

        home_stats = averages[(averages['year'].isin(['2023', '2024'])) & (averages['home'] == True)].groupby('team', as_index=False).mean(numeric_only=True)
        away_stats = averages[(averages['year'].isin(['2023', '2024'])) & (averages['home'] == False)].groupby('team', as_index=False).mean(numeric_only=True)

        to_predict = data.merge(home_stats[['team', 'pts', 'yds', 'turnovers']], left_on='Home Team', right_on='team', how='left')
        to_predict.rename(columns={
            'pts':'pts_per_year_team_a',
            'yds':'team_a_yds',
            'turnovers':'team_a_turnovers'
        }, inplace=True)
        to_predict.drop('team', axis=1, inplace=True)

        to_predict = to_predict.merge(away_stats[['team', 'pts', 'yds', 'turnovers']], left_on='Away Team', right_on='team', how='left')
        to_predict.rename(columns={
            'pts':'pts_per_year_team_b',
            'yds':'team_b_yds',
            'turnovers':'team_b_turnovers'
        }, inplace=True)
        to_predict.drop('team', axis=1, inplace=True)

        model = joblib.load(self.model)

        features = [
            'team_a_yds',
            'team_a_turnovers',
            'team_b_yds',
            'team_b_turnovers',
            'pts_per_year_team_a',
            'pts_per_year_team_b'
        ]

        X = to_predict[features]

        result = model.predict(X)

        to_predict['home_win'] = result
        to_predict = to_predict[['Date', 'Home Team', 'Away Team', 'home_win']]

        winner_labels = []
        for row, value in to_predict.iterrows():
            if value['home_win'] == True:
                winner_labels.append(value['Home Team'])
            else:
                winner_labels.append(value['Away Team'])

        to_predict['Winner'] = winner_labels
        to_predict.drop('home_win', axis=1, inplace=True)
        to_predict['Date'] = to_predict['Date'].astype('datetime64[ns]')
        to_predict.sort_values('Date', inplace=True)
        return to_predict

PredictingKind = PredictingData