import pandas as pd

class ModellingFunctions:
    def __init__(self):
        pass

    def create_pts_per_year(self, df):
        df_pts_per_year = df[['date', 'winner_tie', 'loserttie', 'pts_allowed_winner', 'pts_allowed_loser']]
        df_pts_per_year['year'] = df_pts_per_year['date'].str[:4]

        df_pts_per_year_winner = df_pts_per_year[['year', 'winner_tie', 'pts_allowed_winner']]
        df_pts_per_year_loser = df_pts_per_year[['year', 'loserttie', 'pts_allowed_loser']]

        columns = ['year', 'team', 'pts']
        df_pts_per_year_winner.columns = columns
        df_pts_per_year_loser.columns = columns

        df_pts_per_year = pd.concat([df_pts_per_year_winner, df_pts_per_year_loser])
        df_pts_per_year['pts'] = df_pts_per_year['pts'].astype('int')
        df_pts_per_year = df_pts_per_year.groupby(['year', 'team'], as_index=False).mean().round(2)
        return df_pts_per_year

    def join_pts_per_year(self, df, df_pts_per_year):
        df['year'] = df['date'].str[:4]
        df = df.merge(df_pts_per_year, left_on=['year', 'winner_tie'], right_on=['year', 'team'], how='left')
        df['pts_p_year_winner'] = df['pts']
        df.drop('pts', axis=1, inplace=True)

        df = df.merge(df_pts_per_year, left_on=['year', 'loserttie'], right_on=['year', 'team'], how='left')
        df['pts_p_year_loser'] = df['pts']
        df.drop('pts', axis=1, inplace=True)

        df.drop(['year', 'team_x', 'team_y'], axis=1, inplace=True)
        return df

    def fix_columns(self, df):
        df = df[[
            'date',
            'winner_tie',
            'h_a',
            'loserttie',
            'pts_allowed_winner',
            'pts_allowed_loser',
            'yds_gained',
            'tos',
            'yds_allowed',
            'opp_tos',
            'pts_p_year_winner',
            'pts_p_year_loser'
        ]]

        df.columns = [
            'date',
            'winner_tie',
            'home',
            'loser_tie',
            'pts_winner',
            'pts_loser',
            'yds_winner',
            'turnovers_winner',
            'yds_loser',
            'turnovers_loser',
            'pts_p_year_winner',
            'pts_p_year_loser'
        ]

        df['home'] = df['home'].map({'@':1,'':0})

        int_columns = [
            'pts_winner',
            'pts_loser',
            'yds_winner',
            'turnovers_winner',
            'yds_loser',
            'turnovers_loser',
            'pts_p_year_winner',
            'pts_p_year_loser'
        ]

        for column in int_columns:
            df[column] = df[column].astype('int')
        return df
    
    def fix_home_column(self, df):
        is_home = []

        for row in df['home']:
            if row == 1:
                is_home.append(0)
            else:
                is_home.append(1)

        df['home_'] = is_home
        return df

    def create_outcomes_df(self, df):

        df_home = df[df['home'] == 1]
        df_away = df[df['home'] == 0]

        df_home = df_home[[
            'date',
            'winner_tie',
            'home',
            'loser_tie',
            'pts_winner',
            'pts_loser',
            'yds_winner',
            'turnovers_winner',
            'yds_loser',
            'turnovers_loser',
            'pts_p_year_winner',
            'pts_p_year_loser'
        ]]

        df_away = df_away[[
            'date',
            'loser_tie',
            'home_',
            'winner_tie',
            'pts_loser',
            'pts_winner',
            'yds_loser',
            'turnovers_loser',
            'yds_winner',
            'turnovers_winner',
            'pts_p_year_loser',
            'pts_p_year_winner'
        ]]

        columns = [
            'date',
            'team_a',
            'home',
            'team_b',
            'team_a_pts',
            'team_b_pts',
            'team_a_yds',
            'team_a_turnovers',
            'team_b_yds',
            'team_b_turnovers',
            'pts_per_year_team_a',
            'pts_per_year_team_b'
        ]

        df_home.columns = columns
        df_away.columns = columns

        df_outcome = pd.concat([df_home, df_away])

        df_outcome['outcome'] = df_outcome['team_a_pts'] - df_outcome['team_b_pts']

        outcomes = []
        for row in df_outcome['outcome']:
            if row > 0:
                outcomes.append('team_a')
            elif row < 0:
                outcomes.append('team_b')
            else:
                outcomes.append('tie')
        df_outcome['outcome'] = outcomes
        return df_outcome