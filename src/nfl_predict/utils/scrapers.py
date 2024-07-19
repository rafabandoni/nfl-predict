"""Configs the scrapping file for each scrapping source"""

# %% IMPORTS

import pandas as pd
import typing as T
import pydantic as pdt

import numpy as np
from scipy import stats

from abc import ABC, abstractmethod
from urllib.request import urlopen
from bs4 import BeautifulSoup

# %% Scrapers

class Scraper(ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """"
    Gets data from the web into a dataframe.
    """

    KIND: str

    @abstractmethod
    def scrape(self, url : str, year : int) -> pd.DataFrame:
        """
        Uses bf to get into the url and get data from the given year.

        Parameters:
            url (string): The URL that BF will access.
            year (int): The year of data BF will gatter. 
        """

class NFLScraper(Scraper):
    """
    Gets data from the www.pro-football-reference.com website into a dataframe.
    """

    KIND: T.Literal["NFLScraper"] = "NFLScraper"

    @T.override
    def scrape(self, url : str, year : int) -> pd.DataFrame:
        """
        Gets game to game data from pro footbal website into dataframe.

        Parameters: 
            url (str): The URL to get data from.
            years (list): The list of year as integers to get data from.
        """
        
        url = url.format(year=year)
        html = urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
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
    


ScraperKind = NFLScraper