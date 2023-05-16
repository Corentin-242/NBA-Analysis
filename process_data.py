import time
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy
from scipy import signal
import pickle
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import xarray as xr
# from main import Prepoc_game,Process_game,ProcessSeason
import pickle

if __name__ == '__main__':
    # all_nba_games_id = {}
    # seasons = np.arange(2015,2023)
    # for season in seasons:
    #     all_nba_games_id[season] = {}
    #     game_season = ProcessSeason(season)
    #     games_id = game_season.games_id
    #     all_nba_games_id[season] = set()
    #     for game_id in tqdm(games_id):
    #         url = game_id.split('/')
    #         url = url[-1].split('.')
    #         url_clean = url[0]
    #         all_nba_games_id[season].add(url_clean)
            # game_data = game_season.df[game_season.df['URL'] == game_id]
            # game_data.to_csv(f'./data/{season}/{url_clean}.csv')
            
    # # Store data (serialize)
    # with open('all_nba_games_id.pickle', 'wb') as handle:
    #     pickle.dump(all_nba_games_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    start = time.time()
    # Load data (deserialize)
    # with open('all_nba_games_id.pickle', 'rb') as handle:
    #     unserialized_data = pickle.load(handle)
    # print(unserialized_data[2015])
    df = pd.read_csv(f'./data/2015/201511160SAS.csv')
    print(df)
    end = time.time()
    print(end-start)

   
       