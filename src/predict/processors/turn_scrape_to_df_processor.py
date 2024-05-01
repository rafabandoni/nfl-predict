# %% IMPORT
import pandas as pd
import numpy as np
from scipy import stats

# %% TURN SCRAPED DATA INTO PANDAS DATAFRAME
class TurnScrapeData:
    def convert_data(self, player_data, column_headers):
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
        return winner_pf, loser_pf, percentile, results, spreads