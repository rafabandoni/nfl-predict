# %% IMPORTS
import pandas as pd

from predict.processors.scrape_data_processor import ScrapeDataProcessor
from predict.models.get_data_params import GetDataParams
from predict.processors.turn_scrape_to_df_processor import TurnScrapeData
from predict.processors.create_results_df_processor import CreateResultsDfProcessor

# %% SCRAPE DATA IMPLEMENTATION
class ScrapeData:
    def __init__(self, params: GetDataParams):
        self.params = params
    
    def get_data(self):
        # Iterate Player Data Frame for Each Year Specified
        nfl_df = pd.DataFrame()

        for year in self.params.years:
            
            scrape_data: ScrapeDataProcessor = ScrapeDataProcessor()
            player_data, column_headers = scrape_data.scrape_data(self.params.url_template, year)
            
            turn_scrape_to_df: TurnScrapeData = TurnScrapeData()
            winner_pf, loser_pf, percentile, results, spreads = turn_scrape_to_df.convert_data(player_data, column_headers)

            create_results: CreateResultsDfProcessor = CreateResultsDfProcessor()
            results_df = create_results.create_results_df(results, winner_pf, loser_pf, spreads, percentile)
            
            nfl_df = pd.concat([nfl_df,results_df],axis=0)

        nfl_df = nfl_df.reset_index(drop=True)
        nfl_df.columns = ['week', 'day', 'date', 'time', 'winner_tie', 'h_a', 'loserttie',
                          'pts_allowed_winner', 'pts_allowed_loser', 'yds_gained', 'tos', 'yds_allowed',
                          'opp_tos', 'tm_performance', 'opp_performance', 'spread', 'spread_prcntl']
        return nfl_df

