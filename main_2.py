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

with open('all_nba_games_id.pickle', 'rb') as handle:
    all_nba_games_id = pickle.load(handle)


class Prepoc_game:
    def __init__(self,season, game_id) -> None:
        self.game_id = game_id
        self.df_game = pd.read_csv(f'./data/{season}/{game_id}.csv') ## takes 0.02
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
        
    def max_ecart_is(self):
        #Load a specific game and process it
        df_fin = self.game_data.df_clean
        max_ecart_abs = max(abs(df_fin[df_fin['Time'] < 1600]["HOME_MARGIN"]))
        return(max_ecart_abs)
    
    def good_game_max_ecart_is(self): 
        max_ecart_abs = self.max_ecart_is()
        good_game_bool = self.good_game()
        return(max_ecart_abs,good_game_bool)

class ProcessSeason:

    def __init__(self,season) -> None:
        self.season = season
        self.games_id = all_nba_games_id[season]

    def good_games_season(self):
        n_games = len(self.games_id)
        count = 0
        for game in tqdm(self.games_id, position=0, leave=True):
            count += Process_game(self.season,game).good_game()
        print(count,n_games)
        print(count/n_games)

    def trailing_prob_comeback(self):
        good_game_count = defaultdict(lambda: 0)
        all_game_count = defaultdict(lambda: 0)
        
        for game in tqdm(self.games_id, position=0, leave=True):
            max_ecart_abs,good_game = Process_game(self.season,game).good_game_max_ecart_is()
            good_game_count[max_ecart_abs] += good_game
            all_game_count[max_ecart_abs] += 1

        # ecarts = list(all_game_count.keys())
        # ecarts.sort()
        # for ecart in ecarts:
        #     print(f"If a Team was trailing by maximum {ecart} at any moment before 1600 sec into the game, there is a {good_game_count[ecart]/all_game_count[ecart]} probabilty to go to a tie game after 1600 sec. {good_game_count[ecart]} - {all_game_count[ecart]} ")
        return((good_game_count,all_game_count))
    

    


if __name__=='__main__': 
    # ProcessSeason(2015).good_games_season()
    # for season in all_nba_games_id.keys():
    #     ProcessSeason(season).good_games_season()
    # print(game_data.good_game())
    # game = Process_game(2022,'202302110ATL')
    # print(game.good_game_max_ecart_is())
    # print(game.plot_game())
    
    good_game_count_all = defaultdict(lambda: 0)
    all_game_count_all = defaultdict(lambda: 0)

    for season in tqdm(all_nba_games_id.keys()):
        good_game_count,all_game_count = ProcessSeason(season).trailing_prob_comeback()
        for ecart in good_game_count.keys():
            good_game_count_all[ecart] += good_game_count[ecart]
            all_game_count_all[ecart] += all_game_count[ecart]
    ecarts = list(good_game_count.keys())
    ecarts.sort()
    for ecart in ecarts:
        print(f"If a Team was trailing by maximum {ecart} at any moment before 1600 sec into the game, there is a {good_game_count_all[ecart]/all_game_count_all[ecart]} probabilty to go to a tie game after 1600 sec. {good_game_count_all[ecart]} - {all_game_count_all[ecart]} ")



    
    
    
    

    
    

    
    

    