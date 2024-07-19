"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd

from nfl_predict.utils import scrapers

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


class NFLReader(Reader, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["NFLScrapper"] = "NFLScrapper"

    url: str = "https://www.pro-football-reference.com/years/{year}/games.htm"
    years: list = [2022, 2023]
    scraper: scrapers.ScraperKind = pdt.Field(
        scrapers.NFLScraper(), discriminator="KIND"
    )

    @T.override
    def read(self) -> pd.DataFrame:
        nfl_df = pd.DataFrame()

        for year in self.years:
            results_df = self.scraper.scrape(self.url, year)
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
    

ReaderKind = NFLReader | ParquetReader

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


WriterKind = ParquetWriter