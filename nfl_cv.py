from kaggle.competitions import nflrush
# You can only call make_env() once, so don't lose it!
env = nflrush.make_env()


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from string import punctuation
import re

import keras
import sklearn
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import regularizers, losses
from keras import backend as K

import tensorflow as tf
tf.random.set_seed(1)

keras.backend.set_floatx('float32')

player_features=    ['Flag', 'S','A','Dis']
player_features_min=[     0,  0,   0,    0]
player_features_max=[     1, 10,  15,  1.5]

game_nfeatures = ['QuarterClock', 'GameTime', 'YardsToTouchDown',     'CarrierToTouchDown',     'CarrierToYardLine', 'CarrierToSideLine',      'Distance',  
                   'Temperature', 'Humidity',        'WindSpeed', 'OffenseScoreBeforePlay', 'OffenseScoreAdvantage',
                      'CarrierS', 'CarrierA',       'CarrierDis', 'CarrierOrientation_std',        'CarrierDir_std', 'CarrierHeightInch', 'CarrierWeight', 
                    'CarrierAge', 'Carrier_Xforward1','Carrier_Xforward2','Carrier_Xforward3','Carrier_CB_WR_TE','Carrier_QB_DT_DE','Carrier_FB','Carrier_HB_RB']
game_nfeatures_min=[           0,          0,                  0,                        0,                      -1,                   0,               1, 
                               0,          0,                  0,                        0,                       0,
                               0,          0,                  0,                        0,                       1,                  60,             150, 
                    18,0,0,0,0,0,0,0]
game_nfeatures_max=[         900,       4200,                  1,                        1,                      15,             160/3/2,              40, 
                             100,        100,                 25,                       60,                      60,
                              10,         15,                1.5,                      360,                     360,                  85,             400, 
                    45,1,1,1,1,1,1,1]

game_cfeatures = ['IsHome', 'Quarter_1', 'Quarter_2', 'Quarter_3', 'Quarter_4', 'Quarter_5', 'Down_1', 'Down_2', 'Down_3', 'Down_4',
                  'Turf', 'GameWeather', 'StadiumType']

tile_features = ['IsBallCarrier', 'X_std','Y_std','IsOnOffense','X1','X2','Y1','Y2']
other_features = ['GameId', 'PlayId']

def euclidean_distance(x1,y1,x2,y2):
  x_diff = (x1-x2)**2
  y_diff = (y1-y2)**2
  return np.sqrt(x_diff + y_diff)

def clean_df(df):
    # https://www.kaggle.com/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars
    df['toLeft'] = (df['PlayDirection'] == "left")
    df['IsBallCarrier'] = (df['NflId'] == df['NflIdRusher'])
    
    df.loc[df['VisitorTeamAbbr'] == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df.loc[df['HomeTeamAbbr']    == "ARI", 'HomeTeamAbbr']    = "ARZ"
    df.loc[df['VisitorTeamAbbr'] == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df.loc[df['HomeTeamAbbr']    == "BAL", 'HomeTeamAbbr']    = "BLT"
    df.loc[df['VisitorTeamAbbr'] == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df.loc[df['HomeTeamAbbr']    == "CLE", 'HomeTeamAbbr']    = "CLV"
    df.loc[df['VisitorTeamAbbr'] == "HOU", 'VisitorTeamAbbr'] = "HST"
    df.loc[df['HomeTeamAbbr']    == "HOU", 'HomeTeamAbbr']    = "HST"
    
    # https://www.kaggle.com/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars
    df['TeamOnOffense'] = np.where(df['PossessionTeam']==df['HomeTeamAbbr'], "home", "away")
    df['IsOnOffense'] = (df['Team']==df['TeamOnOffense'])
    df['YardsFromOwnGoal'] = np.where(df['FieldPosition']==df['PossessionTeam'], df['YardLine'], 50 + (50-df['YardLine']))
    df['YardsFromOwnGoal'] =  np.where(df['YardLine']==50, 50, df['YardsFromOwnGoal'])
    df['X_std'] = np.where(df['toLeft'], 120-df['X'], df['X']) - 10
    df['Y_std'] = np.where(df['toLeft'], 160/3-df['Y'], df['Y'])
    
    #https://www.kaggle.com/ben519/understanding-x-y-dir-and-orientation
    df['Orientation_new'] = np.where(df['Season']==2017, np.mod(df['Orientation']+90, 360) , df['Orientation'])
    
    # standardize Dir and Orientation
    df['Dir_std'] = np.where(df['toLeft'], np.mod(df['Dir']+180, 360), df['Dir']) 
    df['Orientation_std'] = np.where(df['toLeft'], np.mod(df['Orientation_new']+180, 360), df['Orientation_new'])
    
    # change data types for some features
    df['QuarterClock']= pd.to_numeric(df['GameClock'].str[:2])*60 + pd.to_numeric(df['GameClock'].str[3:5])
    df['GameTime']= np.where(df['Quarter']==5, \
                             df['Quarter'] * 15*60 - 5*60 - pd.to_numeric(df['GameClock'].str[:2])*60 - pd.to_numeric(df['GameClock'].str[3:5]),\
                             df['Quarter'] * 15*60 - pd.to_numeric(df['GameClock'].str[:2])*60 - pd.to_numeric(df['GameClock'].str[3:5]))
    df['PlayerHeightInch'] = pd.to_numeric(df['PlayerHeight'].str[:1])*12+pd.to_numeric(df['PlayerHeight'].str[2:])
    df['PlayerAge'] = (pd.to_datetime(df['TimeHandoff']).dt.date - pd.to_datetime(df['PlayerBirthDate']).dt.date)/np.timedelta64(365, 'D')
    df['Flag'] = 1
    
    # categorical features and one hot encoding
    #df = pd.concat([df, pd.get_dummies(pd.Categorical(df['Quarter']), prefix = 'Quarter')], axis=1)
    #df = pd.concat([df, pd.get_dummies(pd.Categorical(df['Down']), prefix = 'Down')], axis=1)
    df['Quarter_1']=np.where(df['Quarter']==1, 1, 0)
    df['Quarter_2']=np.where(df['Quarter']==2, 1, 0)
    df['Quarter_3']=np.where(df['Quarter']==3, 1, 0)
    df['Quarter_4']=np.where(df['Quarter']==4, 1, 0)
    df['Quarter_5']=np.where(df['Quarter']==5, 1, 0)
    df['Down_1']=np.where(df['Down']==1, 1, 0)
    df['Down_2']=np.where(df['Down']==2, 1, 0)
    df['Down_3']=np.where(df['Down']==3, 1, 0)
    df['Down_4']=np.where(df['Down']==4, 1, 0)
    
    #https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
    grass = ['grass', 'natural grass', 'natural', 'naturall grass']
    df['Turf'] = np.where(df['Turf'].str.lower().isin(grass), 1, 0)
    
    # https://stackoverflow.com/questions/26577516/how-to-test-if-a-string-contains-one-of-the-substrings-in-a-list-in-pandas
    rain=['rain','shower','snow']
    df['GameWeather'] = np.where(df['GameWeather'].str.lower().str.contains('|'.join(rain)), 1, 0)
    
    # https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    def clean_StadiumType(txt):
        if pd.isna(txt):
            return np.nan
        txt = txt.lower()
        txt = ''.join([c for c in txt if c not in punctuation])
        txt = re.sub(' +', ' ', txt)
        txt = txt.strip()
        txt = txt.replace('outside', 'outdoor')
        txt = txt.replace('outdor', 'outdoor')
        txt = txt.replace('outddors', 'outdoor')
        txt = txt.replace('outdoors', 'outdoor')
        txt = txt.replace('oudoor', 'outdoor')
        txt = txt.replace('indoors', 'indoor')
        txt = txt.replace('ourdoor', 'outdoor')
        txt = txt.replace('retractable', 'rtr.')
        return txt

    def transform_StadiumType(txt):
        if pd.isna(txt):
            return np.nan
        if 'outdoor' in txt or 'open' in txt:
            return 1
        if 'indoor' in txt or 'closed' in txt:
            return 0 
        return np.nan
    
    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
    df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)
    
    # https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    def str_to_float(txt):
        try:
            return float(txt)
        except:
            return np.nan
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    # https://howtheyplay.com/team-sports/Offensive-and-Defensive-Football-Positions-Explained
    # postion coding
    df['QB'] = np.where((df['JerseyNumber']>=1) & (df['JerseyNumber']<=19), 1, 0)
    df['RB'] = np.where((df['JerseyNumber']>=20) & (df['JerseyNumber']<=49), 1, 0)
    df['WR'] = np.where(((df['JerseyNumber']>=10) & (df['JerseyNumber']<=19))  | ((df['JerseyNumber']>=80) & (df['JerseyNumber']<=89)) , 1, 0)
    df['TE'] = np.where(((df['JerseyNumber']>=40) & (df['JerseyNumber']<=49))  | ((df['JerseyNumber']>=80) & (df['JerseyNumber']<=89)) , 1, 0)
    df['DL'] = np.where(((df['JerseyNumber']>=50) & (df['JerseyNumber']<=79))  | ((df['JerseyNumber']>=90) & (df['JerseyNumber']<=99)) , 1, 0)
    df['LB'] = np.where(((df['JerseyNumber']>=40) & (df['JerseyNumber']<=59))  | ((df['JerseyNumber']>=90) & (df['JerseyNumber']<=99)) , 1, 0)
    df['ballon_CB_WR_TE'] = np.where((df['Position']=='CB') | (df['Position']=='WR') | (df['Position']=='TE'), 1, 0)
    df['ballon_QB_DT_DE'] = np.where((df['Position']=='QB') | (df['Position']=='DT') | (df['Position']=='DE'), 1, 0)
    df['ballon_FB'] = np.where((df['Position']=='FB') , 1, 0)
    df['ballon_HB_RB'] = np.where((df['Position']=='HB') | (df['Position']=='RB'), 1, 0)

    # distance after 1/2/3 seconds
    df['Displancement1'] = df['S']+df['A']*0.5
    df['Displancement2'] = df['S']*2+df['A']*0.5*4
    df['Displancement3'] = df['S']*3+df['A']*0.5*9
    # position after 1 or 2 seconds
    df['X1']=df['X_std']+df['Displancement1']*np.sin(df['Dir_std'])
    df['X2']=df['X_std']+df['Displancement2']*np.sin(df['Dir_std'])
    df['Y1']=df['Y_std']+df['Displancement1']*np.cos(df['Dir_std'])
    df['Y2']=df['Y_std']+df['Displancement2']*np.cos(df['Dir_std'])

    # distance from ball carrier
    df['Carrier_X'] = np.where(df['IsBallCarrier'], df['X_std'], np.nan)
    df['Carrier_Y'] = np.where(df['IsBallCarrier'], df['Y_std'], np.nan)
    df["Carrier_X"] = df[['GameId', 'PlayId','Carrier_X']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    df["Carrier_Y"] = df[['GameId', 'PlayId','Carrier_Y']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    df['Dist2Carrier'] = df[['X_std','Y_std','Carrier_X','Carrier_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    df['Dist2Carrier'] = np.where(df['IsOnOffense'], 200, df['Dist2Carrier'])  # only count defense team
    df['X2Carrier'] = df[['X_std', 'Carrier_X']].apply(lambda x: x[0]-x[1], axis=1)
    df['Y2Carrier'] = df[['Y_std', 'Carrier_Y']].apply(lambda x: x[0]-x[1], axis=1)
    df['X2Carrier'] = np.where(np.absolute(df['Y2Carrier'])>2, 200, df['X2Carrier'])  # only count players not too far side 
    df['X2Carrier'] = np.where(df['X2Carrier']<0, 200, df['X2Carrier']) # only count players in front of carrier
    df['X2Carrier'] = np.where(df['IsOnOffense'], 200, df['X2Carrier']) # only count defense team
    df['FrontofCarrier'] = np.where((df['X2Carrier']>0) & (df['X2Carrier']<200) , 1, 0)
    

    # distance from ball carrier after 1 seconds
    #df['Carrier_X1'] = np.where(df['IsBallCarrier'], df['X1'], np.nan)
    #df['Carrier_Y1'] = np.where(df['IsBallCarrier'], df['Y1'], np.nan)
    #df["Carrier_X1"] = df[['GameId', 'PlayId','Carrier_X1']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    #df["Carrier_Y1"] = df[['GameId', 'PlayId','Carrier_Y1']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    #df['Dist2Carrier1'] = df[['X1','Y1','Carrier_X1','Carrier_Y1']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    #df['Dist2Carrier1'] = np.where(df['IsOnOffense'], 200, df['Dist2Carrier1'])  # only count defense team
    #df['X2Carrier1'] = df[['X1', 'Carrier_X1']].apply(lambda x: x[0]-x[1], axis=1)
    #df['Y2Carrier1'] = df[['Y1', 'Carrier_Y1']].apply(lambda x: x[0]-x[1], axis=1)
    #df['X2Carrier1'] = np.where(np.absolute(df['Y2Carrier1'])>2, 200, df['X2Carrier1'])  # only count players not too far side 
    #df['X2Carrier1'] = np.where(df['X2Carrier1']<0, 200, df['X2Carrier1']) # only count players in front of carrier
    #df['X2Carrier1'] = np.where(df['IsOnOffense'], 200, df['X2Carrier1']) # only count defense team
    #df['FrontofCarrier1'] = np.where((df['X2Carrier1']>0) & (df['X2Carrier1']<200) , 1, 0)

    # distance from ball carrier after 2 seconds
    #df['Carrier_X2'] = np.where(df['IsBallCarrier'], df['X2'], np.nan)
    #df['Carrier_Y2'] = np.where(df['IsBallCarrier'], df['Y2'], np.nan)
    #df["Carrier_X2"] = df[['GameId', 'PlayId','Carrier_X2']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    #df["Carrier_Y2"] = df[['GameId', 'PlayId','Carrier_Y2']].groupby(['GameId', 'PlayId']).transform(lambda x: x.fillna(x.max()))
    #df['Dist2Carrier2'] = df[['X2','Y2','Carrier_X2','Carrier_Y2']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    #df['Dist2Carrier2'] = np.where(df['IsOnOffense'], 200, df['Dist2Carrier2'])  # only count defense team
    #df['X2Carrier2'] = df[['X2', 'Carrier_X2']].apply(lambda x: x[0]-x[1], axis=1)
    #df['Y2Carrier2'] = df[['Y2', 'Carrier_Y2']].apply(lambda x: x[0]-x[1], axis=1)
    #df['X2Carrier2'] = np.where(np.absolute(df['Y2Carrier2'])>2, 200, df['X2Carrier2'])  # only count players not too far side 
    #df['X2Carrier2'] = np.where(df['X2Carrier2']<0, 200, df['X2Carrier2']) # only count players in front of carrier
    #df['X2Carrier2'] = np.where(df['IsOnOffense'], 200, df['X2Carrier2']) # only count defense team
    #df['FrontofCarrier2'] = np.where((df['X2Carrier2']>0) & (df['X2Carrier2']<200) , 1, 0)

    df.fillna(0)
    
    return df

def extract_game_feature(df, is_train=True):
    # team features
    
    df['IsHome'] = (df['HomeTeamAbbr']==df['PossessionTeam']).astype(int)
    df['YardsToTouchDown'] = 100 - df['YardsFromOwnGoal']
    
    df['OffenseScoreBeforePlay'] =  np.where(df['TeamOnOffense']=="home", df['HomeScoreBeforePlay'], df['VisitorScoreBeforePlay'])
    df['OffenseScoreAdvantage'] = df['OffenseScoreBeforePlay'] - np.where(df['TeamOnOffense']=="home", df['VisitorScoreBeforePlay'], df['HomeScoreBeforePlay'])
    
    # carrier features
    df['CarrierToTouchDown'] =  np.where(df['IsBallCarrier'], 100 - df['X_std'], 0)
    df['CarrierToYardLine'] =  np.where(df['IsBallCarrier'], df['YardsFromOwnGoal'] - df['X_std'] , -200)
    df['CarrierToSideLine'] =  np.where(df['IsBallCarrier'], np.minimum(160/3-df['Y_std'], df['Y_std']) , 0)
    
    df['CarrierS'] = np.where(df['IsBallCarrier'], df['S'], 0)
    df['CarrierA'] = np.where(df['IsBallCarrier'], df['A'], 0)
    df['CarrierDis'] = np.where(df['IsBallCarrier'], df['Dis'], 0)
    df['CarrierOrientation_std'] = np.where(df['IsBallCarrier'], df['Orientation_std'], 0)
    df['CarrierDir_std'] = np.where(df['IsBallCarrier'], df['Dir_std'], 0)
    df['CarrierHeightInch'] = np.where(df['IsBallCarrier'], df['PlayerHeightInch'], 0)
    df['CarrierWeight'] = np.where(df['IsBallCarrier'], df['PlayerWeight'], 0)
    df['CarrierAge'] = np.where(df['IsBallCarrier'], df['PlayerAge'], 0)

    df['Carrier_CB_WR_TE'] = np.where(df['IsBallCarrier'], df['ballon_CB_WR_TE'], 0)
    df['Carrier_QB_DT_DE'] = np.where(df['IsBallCarrier'], df['ballon_QB_DT_DE'], 0)
    df['Carrier_FB'] = np.where(df['IsBallCarrier'], df['ballon_FB'], 0)
    df['Carrier_HB_RB'] = np.where(df['IsBallCarrier'], df['ballon_HB_RB'], 0)

    df['Carrier_Xforward1'] = np.where(df['IsBallCarrier'], df['Displancement1']*np.sin(df['Dir_std']), -200)
    df['Carrier_Xforward2'] = np.where(df['IsBallCarrier'], df['Displancement2']*np.sin(df['Dir_std']), -200)
    df['Carrier_Xforward3'] = np.where(df['IsBallCarrier'], df['Displancement3']*np.sin(df['Dir_std']), -200)

    df['Close2Carrier1'] = np.where(df['Dist2Carrier']<=1, 1, 0)
    df['Close2Carrier2'] = np.where(df['Dist2Carrier']<=2, 1, 0)

    df_play = df[other_features+game_nfeatures+game_cfeatures].groupby(['GameId', 'PlayId']).max()
    df_play[game_nfeatures] = ((df_play[game_nfeatures]-game_nfeatures_min)/game_nfeatures_max).round(decimals=2)
    df_play[game_nfeatures] = df_play[game_nfeatures].astype('float16')
    df_play['StadiumType'] = df_play['StadiumType'].astype('float16')
    df_play['Close2Carrier1']=df[['GameId', 'PlayId', 'Close2Carrier1']].groupby(['GameId', 'PlayId']).sum()
    df_play['Close2Carrier2']=df[['GameId', 'PlayId', 'Close2Carrier2']].groupby(['GameId', 'PlayId']).sum()
    df_play['FrontofCarrier']=df[['GameId', 'PlayId', 'FrontofCarrier']].groupby(['GameId', 'PlayId']).sum()
    #df_play['FrontofCarrier1']=df[['GameId', 'PlayId', 'FrontofCarrier1']].groupby(['GameId', 'PlayId']).sum()
    #df_play['FrontofCarrier2']=df[['GameId', 'PlayId', 'FrontofCarrier2']].groupby(['GameId', 'PlayId']).sum()
    df_play['X2Carrier_min']=df[['GameId', 'PlayId', 'X2Carrier']].groupby(['GameId', 'PlayId']).min()
    #df_play['X2Carrier1_min']=df[['GameId', 'PlayId', 'X2Carrier1']].groupby(['GameId', 'PlayId']).min()
    #df_play['X2Carrier2_min']=df[['GameId', 'PlayId', 'X2Carrier2']].groupby(['GameId', 'PlayId']).min()

    if is_train:
        y = df[other_features+['Yards']].groupby(['GameId', 'PlayId']).max()
        return df_play, y
    else: 
        return df_play

def one_tile_data(play, X_lim=15, Y_lim=10, breaks=2):
    # extract one tile of the field as a image with features as channels and the ball carrier as center position
    # X_lim, Y_lim: yard distance with the ball carrier
    # break 1 yard into 2 pixels

    ball_carrier_pos = play.loc[play['IsBallCarrier'], ['X_std','Y_std']]
    play_tile = play.loc[(np.absolute(play['X_std']-ball_carrier_pos['X_std'].values)<=X_lim) & (np.absolute(play['Y_std']-ball_carrier_pos['Y_std'].values)<=Y_lim),\
                       player_features+['X_std','Y_std','IsOnOffense']]
    play_tile[player_features] = play_tile[player_features].astype('float16')
    play_tile[player_features] = ((play_tile[player_features]-player_features_min)/player_features_max).round(decimals=2)
    play_tile['Flag'] = play_tile['Flag'].astype('int')
    
    play_tile_offense = play_tile.loc[play_tile['IsOnOffense'],:]
    play_tile_defense = play_tile.loc[~play_tile['IsOnOffense'],:]
    #play_tile1 = play.loc[(np.absolute(play['X1']-ball_carrier_pos['X_std'].values)<=X_lim) & (np.absolute(play['Y1']-ball_carrier_pos['Y_std'].values)<=Y_lim) & \
    #                      (~ play['IsBallCarrier']), \
    #                      ['X1','Y1','IsOnOffense','Flag']]
    #play_tile2 = play.loc[(np.absolute(play['X2']-ball_carrier_pos['X_std'].values)<=X_lim) & (np.absolute(play['Y2']-ball_carrier_pos['Y_std'].values)<=Y_lim) & \
    #                      (~ play['IsBallCarrier']), \
    #                      ['X2','Y2','IsOnOffense','Flag']]
    #play_tile_offense1 = play_tile1.loc[play_tile1['IsOnOffense'],:]
    #play_tile_defense1 = play_tile1.loc[~play_tile1['IsOnOffense'],:]
    #play_tile_offense2 = play_tile2.loc[play_tile2['IsOnOffense'],:]
    #play_tile_defense2 = play_tile2.loc[~play_tile2['IsOnOffense'],:]

    #play_tile_array = np.zeros((X_lim*2*breaks+1, Y_lim*2*breaks+1, len(player_features)*2+4), dtype='float16')
    play_tile_array = np.zeros((X_lim*2*breaks+1, Y_lim*2*breaks+1, len(player_features)*2), dtype='float16')
    play_tile_array[((play_tile_offense['X_std']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
                    ((play_tile_offense['Y_std']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
                    :len(player_features)] = play_tile_offense[player_features]
    play_tile_array[((play_tile_defense['X_std']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
                    ((play_tile_defense['Y_std']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
                    len(player_features):len(player_features)*2] = play_tile_defense[player_features]
    #play_tile_array[((play_tile_offense1['X1']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
    #                ((play_tile_offense1['Y1']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
    #                len(player_features)*2] = play_tile_offense1['Flag']
    #play_tile_array[((play_tile_defense1['X1']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
    #                ((play_tile_defense1['Y1']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
    #                len(player_features)*2+1] = play_tile_defense1['Flag']
    #play_tile_array[((play_tile_offense2['X2']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
    #                ((play_tile_offense2['Y2']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
    #                len(player_features)*2+2] = play_tile_offense2['Flag']
    #play_tile_array[((play_tile_defense2['X2']-ball_carrier_pos['X_std'].values)*breaks).apply(np.ceil).astype(int)+X_lim*breaks, \
    #                ((play_tile_defense2['Y2']-ball_carrier_pos['Y_std'].values)*breaks).apply(np.ceil).astype(int)+Y_lim*breaks, \
    #                len(player_features)*2+3] = play_tile_defense2['Flag']
    play_tile_array = play_tile_array[(10*breaks):,:,:]
    
    return play_tile_array

def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 


# https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        #X_train, y_train = self.data[0][0], self.data[0][1]
        #y_pred = self.model.predict(X_train)
        #y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        #y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        #tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[0].shape[0])
        #logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[0].shape[0])
        logs['val_CRPS'] = val_s
        #print('tr CRPS', tr_s, 'val CRPS', val_s)
        print('val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


# https://github.com/JHart96/keras_ordinal_categorical_crossentropy/blob/master/ordinal_categorical_crossentropy.py
def OCC_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def get_train_valid_data(df, x1, y, game_ids):
#def get_train_valid_data(df, x1, y, game_ids, is_train=True):
  ids, x2_data = [], []
  df_temp = df.loc[np.in1d(df['GameId'], game_ids),:]
  #x1s, ys = extract_game_feature(df_temp, is_train=is_train)
  for name, group in df_temp[tile_features+other_features+player_features].groupby(['GameId', 'PlayId']):
    ids.append(np.asarray(name))
    x2 = one_tile_data(group.copy())
    x2_data.append(x2)
    
  ids, x2_data = np.asarray(ids), np.asarray(x2_data)
  x1_data = x1.droplevel('GameId').loc[ids[:,1]]
  y_data = y.droplevel('GameId').loc[ids[:,1]] 

  return x1_data, x2_data, y_data


df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
df = clean_df(df)
x1s, ys = extract_game_feature(df, is_train=True)

game_ids = np.unique(df['GameId'])
np.random.seed(2020)
#ids_rand = np.random.rand(len(game_ids))
#game_ids_train = game_ids[(ids_rand<0.9)]
#game_ids_test = game_ids[(ids_rand>=0.9)]

#x1s_test, x2s_test, y_test = get_train_valid_data(df, x1s, ys, game_ids_test)

#print(x1s_test.shape, x2s_test.shape, y_test.shape)

#x1s_test[np.isnan(x1s_test)]=0
#x2s_test[np.isnan(x2s_test)]=0
#y1_test = np.zeros((y_test.shape[0], 199), dtype='float16')
#y2_test = np.zeros((y_test.shape[0], 199), dtype='float16')
#for idx, target in enumerate(list(y_test.iloc[:,0])):
#    y1_test[idx][99 + target] = 1
#    y2_test[idx][(99 + target):] = 1
    
    
train_err=[]
valid_err=[]

num_iter = 5

for i in range(num_iter):
  np.random.seed(2021+i)
  df_new=df
  df_new['X_std']=df_new['X_std']+np.random.rand(df_new.shape[0])-0.5
  df_new['Y_std']=df_new['Y_std']+np.random.rand(df_new.shape[0])-0.5
  game_ids_train=game_ids
  ids_rand_cv = np.random.rand(len(game_ids_train))
  game_ids_cv_train = game_ids_train[(ids_rand_cv<0.8)]
  game_ids_cv_valid = game_ids_train[(ids_rand_cv>=0.8)]
  x1s_train, x2s_train, y_train = get_train_valid_data(df_new, x1s, ys, game_ids_cv_train)
  x1s_valid, x2s_valid, y_valid = get_train_valid_data(df, x1s, ys, game_ids_cv_valid)
  x1s_train=  np.concatenate((x1s_train, x1s_train[:5000]))
  y_train  = np.concatenate((y_train, y_train[:5000]))
  x2s_train= np.concatenate((x2s_train, np.flip(x2s_train[:5000],2)))
  
  print(x1s_train.shape, x2s_train.shape, y_train.shape)
  print(x1s_valid.shape, x2s_valid.shape, y_valid.shape)

  x1s_train[np.isnan(x1s_train)]=0
  x1s_valid[np.isnan(x1s_valid)]=0
  x2s_train[np.isnan(x2s_train)]=0
  x2s_valid[np.isnan(x2s_valid)]=0
  
  y1_train = np.zeros((y_train.shape[0], 199), dtype='float16')
  y2_train = np.zeros((y_train.shape[0], 199), dtype='float16')
  for idx, target in enumerate(list(y_train[:,0])):
  #for idx, target in enumerate(list(y_train.iloc[:,0])):
    y1_train[idx][99 + target] = 1
    y2_train[idx][(99 + target):] = 1
	
  y1_valid = np.zeros((y_valid.shape[0], 199), dtype='float16')
  y2_valid = np.zeros((y_valid.shape[0], 199), dtype='float16')
  for idx, target in enumerate(list(y_valid.iloc[:,0])):
    y1_valid[idx][99 + target] = 1
    y2_valid[idx][(99 + target):] = 1

  m1_input = Input(shape=(x1s_train.shape[1],))
  m2_input = Input(shape=x2s_train.shape[1:])
	
  m2 = Conv2D(16, (3, 3), activation='relu')(m2_input)
  m2 = Conv2D(16, (3, 3), activation='relu')(m2)
  m2 = MaxPooling2D(pool_size=(2, 2))(m2)
  m2 = Conv2D(32, (3, 3), activation='relu')(m2)
  m2 = Conv2D(32, (3, 3), activation='relu')(m2)
  m2 = MaxPooling2D(pool_size=(2, 2))(m2)
  m2 = Conv2D(64, (3, 3), activation='relu')(m2)
  m2 = Conv2D(64, (3, 3), activation='relu')(m2)
  m2 = MaxPooling2D(pool_size=(2, 2))(m2)
  m2_output = Flatten()(m2)

  m1 = Dense(32, activation="relu")(m1_input)
  m1 = Dense(32, activation="relu")(m1)

  merged_model = concatenate([m1, m2_output])
  merged_model = Dense(64, activation="relu")(merged_model)
  merged_model = Dense(64, activation="relu")(merged_model)

  predictions = Dense(199, activation='softmax')(merged_model)
  model = Model(inputs=[m1_input, m2_input], outputs=predictions)
  #model.summary()
  
  adam = Adam(lr=0.001)
  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[])
  es = EarlyStopping(monitor='val_CRPS', mode='min', restore_best_weights=True, 
                     verbose=0,  patience=5)
  es.set_model(model)
  metric = Metric(model, [es], [([x1s_train, x2s_train],y1_train), ([x1s_valid, x2s_valid],y1_valid)])
  history = model.fit([x1s_train, x2s_train], y1_train,batch_size=64, epochs=50, 
                      validation_data = ([x1s_valid, x2s_valid],y1_valid),
                      callbacks=[metric], verbose=0, shuffle=True)
  
  train_err.append(crps(y1_train, model.predict([x1s_train, x2s_train])))
  valid_err.append(crps(y1_valid, model.predict([x1s_valid, x2s_valid])))
  model.save('NFL'+str(i)+'.h5')


print(np.mean(valid_err))

from keras.models import load_model

iter_test = env.iter_test()


models = []
for i in range(num_iter):
    models.append(load_model('NFL'+str(i)+'.h5', custom_objects={'OCC_loss':OCC_loss}))


for (test_df, sample_prediction_df) in iter_test:
    test_df = clean_df(test_df)
    x1_test = extract_game_feature(test_df, is_train=False)
    x2_test = one_tile_data(test_df)
    #x1_test = np.expand_dims(x1_test, axis=0)
    x2_test = np.expand_dims(x2_test, axis=0)
    x1_test[np.isnan(x1_test)]=0
    x2_test[np.isnan(x2_test)]=0
    
    y_pred = np.mean([model.predict([x1_test, x2_test]) for model in models], axis=0)
    
    #https://www.kaggle.com/coolcoder22/nfl-001-neural-network
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)

env.write_submission_file()