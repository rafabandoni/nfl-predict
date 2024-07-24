"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
import numpy as np
import requests
import joblib

from scipy import stats
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup

# %% READERS

class Reader(ABC, pdt.BaseModel, strict=True, frozen=True, extra='forbid'):
    """
    Base class to read data from dataset.

    Used to get data from the dataset and load as dataframe into memory.

    Parameters:
        limit (int, optional): maximum number of rows to read. Defautls to none.
    """
    
    KIND: str

    limit: int | None = None

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """"
        Reads a dataframe from a dataset.

        Returns:
            pd.Dataframe: dataframe representation.
        """


class Scrapper(Reader, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["Scrapper"] = "Scrapper"

    url: str
    years: list

    def scrape(self, year, url):
        url = url.format(year=year)
        # html = urlopen(url)
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
        column_headers = [th.getText() for th in soup.findAll('thead', limit=1)[0].findAll('th')]
        data_rows = soup.findAll('tbody', limit=1)[0].findAll('tr')[0:]
        player_data = [[td.getText() for td in data_rows[i].findAll(['th','td'])] for i in range(len(data_rows))]

        year_df = pd.DataFrame(player_data, columns=column_headers)
        year_df = year_df[(year_df.Date !='Playoffs') & (year_df.Week !='WildCard') & (year_df.Week != 'Week' )]
        year_df = year_df.infer_objects().reset_index(drop=True)
        results = year_df.loc[(year_df.iloc[:,8] != '')]
        winner_pts = np.array(results.iloc[:,8],dtype=int)
        loser_pts = np.array(results.iloc[:,9],dtype=int)
        spreads = winner_pts-loser_pts
        winner_pf = stats.zscore(winner_pts)
        loser_pf = stats.zscore(loser_pts)
        percentile = np.array([stats.norm.cdf(x) for x in stats.zscore(spreads)])

        results_df = pd.concat([results,
                     pd.DataFrame(data=winner_pf,columns=['Winner Performance']),
                     pd.DataFrame(data=loser_pf,columns=['Loser Performance']),
                     pd.DataFrame(data=spreads,columns=['Spread']),
                     pd.DataFrame(data=percentile,columns=['Spread Prcntl']),
                     ],axis=1)
            
        col_len = [x for x in range(results_df.shape[1])]
        col_len.remove(7)
        results_df = results_df.iloc[:,col_len]
        results_df = results_df.rename(columns={list(results_df)[5]:'H/A',
                                list(results_df)[7]:'Pts Scored',
                                list(results_df)[8]:'Pts Allowed',
                                list(results_df)[9]:'Yds Gained',
                                list(results_df)[10]:'TOs',
                                list(results_df)[11]:'Yds Allowed',
                                list(results_df)[12]:'Opp TOs',
                                list(results_df)[13]:'Tm Performance',
                                list(results_df)[14]:'Opp Performance'})
        return results_df

    @T.override
    def read(self) -> pd.DataFrame:
        nfl_df = pd.DataFrame()

        for year in self.years:
            results_df = self.scrape(year, self.url)
            nfl_df = pd.concat([nfl_df,results_df],axis=0)

        nfl_df = nfl_df.reset_index(drop=True)
        nfl_df.columns = ['week', 'day', 'date', 'time', 'winner_tie', 'h_a', 'loserttie',
                          'pts_allowed_winner', 'pts_allowed_loser', 'yds_gained', 'tos', 'yds_allowed',
                          'opp_tos', 'tm_performance', 'opp_performance', 'spread', 'spread_prcntl']
        if self.limit is not None:
            nfl_df = nfl_df.head(self.limit)
        return nfl_df
    

class ParquetReader(Reader):
    """
    Read a dataframe from a parquet file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["ParquetReader"] = "ParquetReader"

    path: str

    @T.override
    def read(self) -> pd.DataFrame:
        data = pd.read_parquet(self.path)
        if self.limit is not None:
            data = data.head(self.limit)
        return data
    

ReaderKind = Scrapper | ParquetReader

#%% WRITERS

class Writer(ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """
    Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    """

    KIND: str

    @abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """
        Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Writer a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str

    @T.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_parquet(data, self.path)

class ModelWriter(Writer):
    """
    TODO
    """

    KIND: T.Literal["ModelWriter"] = "ModelWriter"

    path: str

    @T.override
    def write(self, model) -> None:
        joblib.dump(model, self.path) 


WriterKind = ParquetWriter | ModelWriter