# %% IMPORTS
import os

from predict.models.get_data_params import GetDataParams
from predict.jobs.get_data_impl import ScrapeData

# %% Get Data
def get_data():
    params = GetDataParams(
        url_template="https://www.pro-football-reference.com/years/{year}/games.htm",
        years=[2022, 2023]
    )

    scrape = ScrapeData(params=params)
    try:
        df = scrape.get_data()
    except Exception as e:
        print(e)
    else:
        data_dir = os.path.join(os.getcwd(), 'data')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        df.to_parquet(os.path.join(data_dir, 'nfl_data.parquet'))