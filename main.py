from nfl_predict.models.get_data_params import GetDataParams
from nfl_predict.get_data import ScrapeData

def main():
    params = GetDataParams(
        url_template="https://www.pro-football-reference.com/years/{year}/games.htm",
        years=[2023]
    )

    scrape = ScrapeData(params=params)
    try:
        df = scrape.get_data()
        # print(df.columns)
    except Exception as e:
        print(e)
    else:
        df.to_parquet('data/nfl_data.parquet')

if __name__ == '__main__':
    main()