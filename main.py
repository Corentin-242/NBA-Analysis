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

#Load every game
# filepaths = {
#             2015 : './data/NBA_PBP_2015-16.csv',
#             2016: './data/NBA_PBP_2016-17.csv',
#             2017: './data/NBA_PBP_2017-18.csv',
#             2018: './data/NBA_PBP_2018-19.csv',
#             2019: './data/NBA_PBP_2019-20.csv',
#             2020: './data/NBA_PBP_2020-21.csv',
#             2021: './data/NBA_PBP_2021-22.csv',
#             2022: './data/NBA_PBP_2022-23.csv'
#                 }
filepaths = {
            2015 : './data/NBA_PBP_2015-16.csv',
            2016: './data/NBA_PBP_2016-17.csv',
            2017: './data/NBA_PBP_2017-18.csv',
            2018: './data/NBA_PBP_2018-19.csv',
            2019: './data/NBA_PBP_2019-20.csv',
            2020: './data/NBA_PBP_2020-21.csv',
            2021: './data/NBA_PBP_2021-22.csv',
            2022: './data/NBA_PBP_2022-23.csv'
                }

all_games_data_pbp = {}
for key,value in tqdm(filepaths.items()):
        print(f"Data from {key} is loading...")
        all_games_data_pbp[key] = pd.read_csv(value)
        

class Prepoc_game:
    def __init__(self,season, game_id) -> None:
        self.df = all_games_data_pbp[season]
        self.game_id = game_id
        self.df_game = self.df[self.df['URL'] == self.game_id] ## takes 0.02
        self.create_df_time() 
        self.df_clean = self.prepare_df() ##takes 0.004

    def create_df_time(self):
        try:
            self.df_time = pd.read_csv('./data/preprocess_data/df_time.csv')
        except FileNotFoundError:
            whole_game = np.arange(0,2881).reshape(-1,1)
            first_quarter = np.arange(0,721)[::-1]
            quarter = np.arange(0,720)[::-1]
            all_quarter = np.concatenate([first_quarter,quarter,quarter,quarter]).reshape(-1,1)
            period = np.concatenate([[1]*721, [2]*720, [3]*720, [4]*720]).reshape(-1,1)
            data = np.concatenate([whole_game,period,all_quarter], axis = 1) 
            self.df_time = pd.DataFrame(data = data,columns = ['Time', 'Quarter', 'SecLeft'])
            self.df_time.to_csv('./data/preprocess_data/df_time.csv',index=False)

    def create_ot_df(self,x):
        plus_time = np.arange(2881+(x-5)*300,2881+(x-4)*300).reshape(-1,1)
        ot_quarter = np.arange(0,300)[::-1].reshape(-1,1).reshape(-1,1)
        period_ot = np.array([x]*300).reshape(-1,1)
        data_ot = np.concatenate([plus_time,period_ot,ot_quarter], axis = 1)
        df_time_ot = pd.DataFrame(data = data_ot,columns = ['Time', 'Quarter', 'SecLeft'])
        return(df_time_ot)
    
    
    
    def prepare_df(self):
        df_temp = self.df_game
        #Filtering and creating interesting columns
        df_temp = df_temp[['Quarter','SecLeft','HomeScore','AwayScore']].drop_duplicates()
        df_temp['HOME_MARGIN'] = df_temp["HomeScore"] - df_temp['AwayScore'] 
        df_temp['Min'] = df_temp["SecLeft"]//60
        df_aux = self.df_time
        
        possible_ot = len(self.df_game['Quarter'].unique())
        if possible_ot > 4:
            for i in range(4,possible_ot):
                df_aux = pd.concat([df_aux, self.create_ot_df(i+1)])
        df_fin = pd.merge(df_aux, df_temp, how="right", on=["Quarter", "SecLeft"])
        return(df_fin)
    
class Process_game:
    def __init__(self,season,game_id) -> None:
        self.game_data = Prepoc_game(season,game_id)
        

    def game_name(self):
        df_temp = self.game_data.df_game  
        hometeam, awayteam = df_temp['HomeTeam'].unique()[0],df_temp['AwayTeam'].unique()[0]
        return(hometeam, awayteam)
    
    def plot_game(self):
        home, away = self.game_name()
        data_to_plot = self.game_data.df_clean
        fig = px.line(data_to_plot, x='Time', y="HOME_MARGIN", title=f'{home} VS {away}')
        return(fig.show())

    def good_game(self): 
        
        df_fin = self.game_data.df_clean
        
        data = df_fin[df_fin['Time'] > 1600]["HOME_MARGIN"]
        sign = np.sign(data).diff().ne(0)
        
        if any(sign[1:]):
            return(1)
        else:
            return(0)

class ProcessSeason:

    def __init__(self,season) -> None:
        self.season = season
        self.df = all_games_data_pbp[season]
        self.games_id = self.df['URL'].unique()

    def good_games_season(self):
        n_games = len(self.games_id)
        count = 0
        for game in tqdm(self.games_id, position=0, leave=True):
            count += Process_game(2022,game).good_game()
        print(count,n_games)
        print(count/n_games)
    

    


if __name__=='__main__': 
    start = time.time()
    game_data = Process_game(2022,'boxscore/202303220UTA.html')
    end = time.time()
    print(end-start)
    # print(game_data.good_game())
    
    

    
    

    
    

    