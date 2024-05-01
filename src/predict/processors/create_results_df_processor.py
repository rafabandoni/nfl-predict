# %% IMPORTS
import pandas as pd

# %% CREATE RESULTS AS DATAFRAME
class CreateResultsDfProcessor:
    def create_results_df(self, results, winner_pf, loser_pf, spreads, percentile):
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