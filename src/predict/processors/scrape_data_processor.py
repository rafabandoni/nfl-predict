# %% IMPORTS
from urllib.request import urlopen
from bs4 import BeautifulSoup

# %% SCRAPE DATA FROM URL
class ScrapeDataProcessor:
    def scrape_data(self, url_template, year):
        url = url_template.format(year=year)  # get the url
        html = urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')
        column_headers = [th.getText() for th in soup.findAll('thead', limit=1)[0].findAll('th')]
        data_rows = soup.findAll('tbody', limit=1)[0].findAll('tr')[0:]
        player_data = [[td.getText() for td in data_rows[i].findAll(['th','td'])] for i in range(len(data_rows))]
        return player_data, column_headers