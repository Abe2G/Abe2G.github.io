
## Working with English Premier League Seasonal Data: Predicting winning team

This is my second ML work for soccer data which is aimed for working with data from multiple source and comparing ML algorithms (Linear Regressor, Boosting (RF, XGB and LGB), Tree based (DT), and SVM) for predicting winning team given team history.

### Data Descreption

This dataset contains data for last 10 seasons (2009 to 2019) of English Premier League including current (2018/19) season. The dataset is sourced from https://datahub.io/sports-data/english-premier-league website and contains various statistical data such as final and half time result, corners, yellow and red cards etc in addition to team information.


```python
#required packages
import pandas as pd ## for data reading and processing
import os ## for OS level file processing
import matplotlib.pyplot as plt  ## for plotting data
import seaborn as sns ## another library to visualize data features
import numpy as np ## for numerical array processing
```


```python
##reading data
DATA_DIR='archive/'
season_files=os.listdir(DATA_DIR)
season_files
```




    ['season-0910.csv',
     'season-1011.csv',
     'season-1112.csv',
     'season-1213.csv',
     'season-1314.csv',
     'season-1415.csv',
     'season-1516.csv',
     'season-1617.csv',
     'season-1718.csv',
     'season-1819.csv']



As we can see, we have total of 10 season data files. This usually pracctical problem that we face when working with realworld project as data may stored over several data stores. Lets merge each file to one DataFrame for ease processing.


```python
all_season_df=[]
for file in season_files:
    season=pd.read_csv(os.path.join(DATA_DIR,file))
    all_season_df.append(season)
league_data=pd.concat(all_season_df,sort=False) #concatinate each dataframe from list by appending to end of dataframe
league_data.head()#print the first five rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>BbMxAHH</th>
      <th>BbAvAHH</th>
      <th>BbMxAHA</th>
      <th>BbAvAHA</th>
      <th>PSH</th>
      <th>PSD</th>
      <th>PSA</th>
      <th>PSCH</th>
      <th>PSCD</th>
      <th>PSCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0</td>
      <td>2009-08-15</td>
      <td>Aston Villa</td>
      <td>Wigan</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>...</td>
      <td>1.28</td>
      <td>1.22</td>
      <td>4.40</td>
      <td>3.99</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E0</td>
      <td>2009-08-15</td>
      <td>Blackburn</td>
      <td>Man City</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>...</td>
      <td>2.58</td>
      <td>2.38</td>
      <td>1.60</td>
      <td>1.54</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E0</td>
      <td>2009-08-15</td>
      <td>Bolton</td>
      <td>Sunderland</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>...</td>
      <td>1.68</td>
      <td>1.61</td>
      <td>2.33</td>
      <td>2.23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E0</td>
      <td>2009-08-15</td>
      <td>Chelsea</td>
      <td>Hull</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>H</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>D</td>
      <td>...</td>
      <td>1.03</td>
      <td>1.02</td>
      <td>17.05</td>
      <td>12.96</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E0</td>
      <td>2009-08-15</td>
      <td>Everton</td>
      <td>Arsenal</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>A</td>
      <td>...</td>
      <td>2.27</td>
      <td>2.20</td>
      <td>1.73</td>
      <td>1.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 77 columns</p>
</div>




```python
league_data.tail() #the last 5 rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>BbMxAHH</th>
      <th>BbAvAHH</th>
      <th>BbMxAHA</th>
      <th>BbAvAHA</th>
      <th>PSH</th>
      <th>PSD</th>
      <th>PSA</th>
      <th>PSCH</th>
      <th>PSCD</th>
      <th>PSCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>375</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Liverpool</td>
      <td>Wolves</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>H</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>H</td>
      <td>...</td>
      <td>1.98</td>
      <td>1.91</td>
      <td>2.01</td>
      <td>1.95</td>
      <td>1.31</td>
      <td>5.77</td>
      <td>10.54</td>
      <td>1.32</td>
      <td>5.89</td>
      <td>9.48</td>
    </tr>
    <tr>
      <th>376</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>...</td>
      <td>2.52</td>
      <td>2.32</td>
      <td>1.72</td>
      <td>1.64</td>
      <td>1.28</td>
      <td>6.33</td>
      <td>10.21</td>
      <td>1.30</td>
      <td>6.06</td>
      <td>9.71</td>
    </tr>
    <tr>
      <th>377</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>D</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>H</td>
      <td>...</td>
      <td>2.27</td>
      <td>2.16</td>
      <td>1.80</td>
      <td>1.73</td>
      <td>1.44</td>
      <td>4.83</td>
      <td>7.62</td>
      <td>1.37</td>
      <td>5.36</td>
      <td>8.49</td>
    </tr>
    <tr>
      <th>378</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>D</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>H</td>
      <td>...</td>
      <td>2.13</td>
      <td>2.08</td>
      <td>1.85</td>
      <td>1.80</td>
      <td>2.10</td>
      <td>3.64</td>
      <td>3.64</td>
      <td>1.91</td>
      <td>3.81</td>
      <td>4.15</td>
    </tr>
    <tr>
      <th>379</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Watford</td>
      <td>West Ham</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>A</td>
      <td>...</td>
      <td>2.25</td>
      <td>2.19</td>
      <td>1.78</td>
      <td>1.72</td>
      <td>2.20</td>
      <td>3.85</td>
      <td>3.21</td>
      <td>2.11</td>
      <td>3.86</td>
      <td>3.41</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 77 columns</p>
</div>




```python
league_data.shape
```




    (3801, 77)




```python
#After merging we have total of 3801 data points with 77 features. Lets inspect all features
league_data.columns
```




    Index(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG',
           'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC',
           'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD',
           'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA',
           'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH',
           'VCD', 'VCA', 'BSH', 'BSD', 'BSA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD',
           'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5',
           'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
           'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA'],
          dtype='object')




```python
league_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3801 entries, 0 to 379
    Data columns (total 77 columns):
    Div         3800 non-null object
    Date        3800 non-null object
    HomeTeam    3800 non-null object
    AwayTeam    3800 non-null object
    FTHG        3800 non-null float64
    FTAG        3800 non-null float64
    FTR         3800 non-null object
    HTHG        3800 non-null float64
    HTAG        3800 non-null float64
    HTR         3800 non-null object
    Referee     3800 non-null object
    HS          3800 non-null float64
    AS          3800 non-null float64
    HST         3800 non-null float64
    AST         3800 non-null float64
    HF          3800 non-null float64
    AF          3800 non-null float64
    HC          3800 non-null float64
    AC          3800 non-null float64
    HY          3800 non-null float64
    AY          3800 non-null float64
    HR          3800 non-null float64
    AR          3800 non-null float64
    B365H       3800 non-null float64
    B365D       3800 non-null float64
    B365A       3800 non-null float64
    BWH         3799 non-null float64
    BWD         3799 non-null float64
    BWA         3799 non-null float64
    GBH         1519 non-null float64
    GBD         1519 non-null float64
    GBA         1519 non-null float64
    IWH         3799 non-null float64
    IWD         3799 non-null float64
    IWA         3799 non-null float64
    LBH         3419 non-null float64
    LBD         3419 non-null float64
    LBA         3419 non-null float64
    SBH         1140 non-null float64
    SBD         1140 non-null float64
    SBA         1140 non-null float64
    WHH         3800 non-null float64
    WHD         3800 non-null float64
    WHA         3800 non-null float64
    SJH         1940 non-null float64
    SJD         1940 non-null float64
    SJA         1940 non-null float64
    VCH         3800 non-null float64
    VCD         3800 non-null float64
    VCA         3800 non-null float64
    BSH         1520 non-null float64
    BSD         1520 non-null float64
    BSA         1520 non-null float64
    Bb1X2       3800 non-null float64
    BbMxH       3800 non-null float64
    BbAvH       3800 non-null float64
    BbMxD       3800 non-null float64
    BbAvD       3800 non-null float64
    BbMxA       3800 non-null float64
    BbAvA       3800 non-null float64
    BbOU        3800 non-null float64
    BbMx>2.5    3800 non-null float64
    BbAv>2.5    3800 non-null float64
    BbMx<2.5    3800 non-null float64
    BbAv<2.5    3800 non-null float64
    BbAH        3790 non-null float64
    BbAHh       3790 non-null float64
    BbMxAHH     3790 non-null float64
    BbAvAHH     3790 non-null float64
    BbMxAHA     3790 non-null float64
    BbAvAHA     3790 non-null float64
    PSH         2660 non-null float64
    PSD         2660 non-null float64
    PSA         2660 non-null float64
    PSCH        2660 non-null float64
    PSCD        2660 non-null float64
    PSCA        2660 non-null float64
    dtypes: float64(70), object(7)
    memory usage: 2.3+ MB
    

Categorical variables: Div, Home_Team, Away_Team, FTR, HTR, Referee
<br>Except Date (should be casted to Date type), other all are continous values.

N.B: Descreption of each features is included <a href='column descreption.txt'>here</a>.


```python
league_data['Date']=league_data['Date'].astype('datetime64[ns]') #casting date value from string to Date
```


```python
#The info() method above also shows number of non null features for all variables. lets drop columns with more than 50% values missed
data_clean = league_data[[column for column in league_data if league_data[column].count() / len(league_data) >= 0.5]]
print("List of dropped columns:", end=" ")
for c in league_data.columns:
    if c not in data_clean.columns:
        print(c, end=", ")
print('\n')
league_data = data_clean
```

    List of dropped columns: GBH, GBD, GBA, SBH, SBD, SBA, BSH, BSD, BSA, 
    
    


```python
league_data.shape
```




    (3801, 68)




```python
# corr = league_data.corr()
# fig = plt.figure(figsize=(25,25))
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(league_data.columns),1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(league_data.columns)
# ax.set_yticklabels(league_data.columns)
# plt.show()
```

From the above correlation matrix when there is no correlation between 2 variables (when correlation is 0 or near 0) the color is gray. The darkest red means there is a perfect positive correlation, while the darkest blue means there is a perfect negative correlation. The matrix gives as interesting focus to drop or retain features and which features has great impact and we can see that score related featured are highly correlated as expected. Features such as 'Div','BbAvAHA','PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA' has significant impact.


```python
#Lets remove features with no significance
del_col_list = ['Div','BbAvAHA','PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA']
league_data=league_data.drop(del_col_list, axis=1)
league_data.shape
```




    (3801, 60)




```python
league_data.columns
```




    Index(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
           'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
           'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
           'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'WHH', 'WHD', 'WHA', 'SJH',
           'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD',
           'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5',
           'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA'],
          dtype='object')



One of big challenge for data science project is identifying which features are best working for prediction. With classical ML technique that we are using and experieanced with, we pre-model the environment and the model tries to predict for unseen data without contextualizing current condition of the environment. This issues becomes one of hot research area in ML to bring conciousiness and change in features. So, at the time of modeling, selecting for best fitting feature from data points or generating new feature from existing features is vital process and known as Feature Engineering. Next I will try to generate new features that support for the prediction of winning team. Understanding the history (no. of wins, drawn, loses, fauls registered, shoots, red and yellow cards shown, availability of particular player, referee, staduim, etc) of teams based on match or season may be taken as factors for wining. However, such important informations are not easily extracted from the dataset and we need to generate using existing features.

Goal difference is counted as the number of goals scored by a team in all league matches across the season, minus the number of goals conceded. If two or more teams finish level on points the team with the better goal difference will finish higher. If two or more teams have the same points and the same goal difference, the team which has scored the higher number of goals will finish higher. General criterias applied commonly to identity champion team are as follows:
<ul><li>Head-to-head points between tied teams</li>
<li>Head-to-head goal difference between tied teams</li>
<li>Goals scored in head-to-head matches among tied teams</li>
<li>Goal difference in all group matches</li>
<li>Goals scored in all group matches</li>
<li>Away goals scored in all group matches</li>
<li>Wins in all group matches</li>
<li>Away wins in all group matches</li>
<li>Disciplinary points (red card = 3 points, yellow card = 1 point, expulsion for two yellow cards in one match = 3 points)</li></ul>


```python
print(league_data.shape)

# for index,game in league_data.iterrows():
#         print(index, game['HomeTeam'])
```

    (3801, 77)
    


```python
# for dt in league_data.Date.tolist():
#     print(getRankings(dt,league_data))
    
# featured_league_dataset=league_data[['FTR']]
# # print(featured_league_dataset.shape)
# seasonal_data= [] 
# for i in range(len(all_season_df)):
#     seasonal_data.append(all_season_df[i][['HomeTeam','AwayTeam','Date','FTHG', 'FTAG', 'FTR']])
# # # print(seasonal_data[2])
# def get_seasonal_history(date,seasonal_match_data):
#     seasonal_status=dict()
#     for index,game in match_data.iterrows():
#             if game['Date']> date:
#                     break
#             # Since, FTR is gold label it should have valid value
#             if game['FTR'] is np.nan:
#                 break
#             home = game['HomeTeam']
#             away = game['AwayTeam']
#             if home not in seasonal_status:
#                 seasonal_status[home] = {
#                     'match_played': 0,
#                     'points': 0,
#                     'win': 0
#                     'drawn':0
#                     'lost':0
#                     'GD':0
#                     'Goals':0
#                 }
#             if away not in seasonal_status:
#                 seasonal_status[away] = {
#                     'match_played': 0,
#                     'points': 0,
#                     'win': 0
#                     'drawn':0
#                     'lost':0
#                     'GD':0
#                     'Goals':0
#                 }

#             seasonal_status[home]['match_played'] += 1
#             seasonal_status[away]['match_played'] += 1
#             match_goal_diff = game['FTHG'] - game['FTAG']
#             seasonal_status[home]['goal_diff'] += match_goal_diff
#             seasonal_status[away]['goal_diff'] -= match_goal_diff
#             if game['FTR'] == 'H':
#                 seasonal_status[home]['points'] += 3
#                 seasonal_status[home]['win'] += 1
#                 seasonal_status[away]['lost'] += 1
#             elif game['FTR'] == 'A':
#                 seasonal_status[away]['points'] += 3
#                 seasonal_status[away]['win'] += 1
#                 seasonal_status[home]['lost'] += 1
#             else:
#                 seasonal_status[home]['points'] += 1
#                 seasonal_status[away]['points'] += 1
#                 seasonal_status[away]['drawn'] += 1
#                 seasonal_status[home]['drawn'] += 1

#     Team = sorted(scores, key=lambda k: scores[k]['points'], reverse=True)
#     Points, Goal_Diff, Win_Rate = [], [], []
#     for name in Team:
#         val = scores[name]
#         Points.append(val['points'])
#         Goal_Diff.append(val['goal_diff'])
#         Win_Rate.append(val['win'] / val['match_played'])
#     df = pd.DataFrame(list(zip(Team, Points, Goal_Diff, Win_Rate)), columns=['Team', 'Points', 'Goal_Diff', 'Win_Rate'])
    
#     return seasonal_status
    
def feature_generation():
  
    for index,game in league_data.iterrows():
        print(index, game['HomeTeam'])
#     count=0
#     for match in seasonal_data:
#         for index,game in match_data.iterrows():
#             if game['Date']> date:
#             break
#         home=match.HomeTeam.tolist()
#         away=match.AwayTeam.tolist()
#         c=[]
#         for h,a in zip(home,away):
#             if h+a not in c:
#                 c.append(h+a)
#             else:
#                 print(a,h)
#         print(len(c))
# #     year=data.Date.dt.year.tolist()
    
# #     ##status of home team by season
# #     SEASON_HOME_NO_OF_WIN- SHW
# #     SEASON_HOME_NO_OF_DRAWN- SHD
# #     SEASON_HOME_NO_OF_LOSE- SHL
# #     SEASON_AWAY_NO_OF_WIN- SAW
# #     SEASON_AWAY_NO_OF_DRAWN- SAD
# #     SEASON_AWAY_NO_OF_LOSE-SAL
    
# #     ##match history of teams
# #     MATCH_HOME_NO_OF_WIN
# #     MATCH_HOME_NO_OF_DRAWN
# #     MATCH_HOME_NO_OF_LOSE
# #     MATCH_AWAY_NO_OF_WIN
# #     MATCH_AWAY_NO_OF_DRAWN
# #     MATCH_AWAY_NO_OF_LOSE
# #     MATCH_HOME_NO_OF_RED
# #     MATCH_HOME_NO_OF_YELLOW
# #     MATCH_HOME_NO_OF_FAULS
# #     MATCH_AWAY_NO_OF_FAULS
# #     MATCH_HOME_NO_OF_SHOOTS
# #     MATCH_AWAY_NO_OF_SHOOTS
# #     MATCH_HOME_NO_OF_TARGET_SHOOTS
# #     MATCH_AWAY_NO_OF_TARGET_SHOOTS
# #     TOTAL_SCORE_AWAY
# #     TOTAL_SCORE_HOME
    
    
# #     matches_15_days
    
# #     ## some times referee maters for probability of lose, drawn and win
# #     Referee
    
    
feature_generation()

```


```python
# data=league_data.groupby('HomeTeam')['FTR'].value_counts()
# data.unstack()
```


```python
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from time import time 
from sklearn.metrics import f1_score
from sklearn.externals import joblib
```


```python

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, labels=['H','D','A'], average = None), sum(target == y_pred) / float(len(y_pred)), clf.score(features, target), y_pred


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc, confidence, _ = predict_labels(clf, X_train, y_train)
#    print("F1 score and accuracy score for training set: {} , {}.".format(f1 , acc))
#    print("Confidence score for training set: {}.".format(confidence))
    
    f1, acc, confidence, predictions = predict_labels(clf, X_test, y_test)
#    print("F1 score and accuracy score for test set: {} , {}.".format(f1 , acc))
    print("Confidence score for test set: {}.".format(confidence))
    print()
    
    return confidence, predictions
    

def get_grid_clf(clf, scoring, param, X_all, y_all):
    gridsearch = GridSearchCV(clf, 
                              scoring=scoring, 
                              param_grid=param, 
                              verbose=100)
    grid_obj = gridsearch.fit(X_all,y_all)
    
    clf = grid_obj.best_estimator_
    params = grid_obj.best_params_
    print(clf)
    print(params)
    
    return clf


def get_random_clf(clf, scoring, param, X_all, y_all):
    randomsearch = RandomizedSearchCV(clf, param, 
                                      n_iter=10,
                                      scoring=scoring,
                                      verbose=100)
    random_obj = randomsearch.fit(X_all,y_all)
    
    clf = random_obj.best_estimator_
    params = random_obj.best_params_
    print(clf)
    print(params)
    
    return clf


def process_print_result(clfs, res):
    def average(lst):
        return sum(lst) / len(lst)
    
    avg_dict = {}
    best_clf_so_far = 0
    best_avg_so_far = -1
    for i in range(len(clfs)):
        clf_name = clfs[i].__class__.__name__
        if clf_name in avg_dict:
            clf_name += json.dumps(clfs[i].get_params())
        avg = average(res[i])
        avg_dict[clf_name] = avg
        if avg > best_avg_so_far:
        	best_avg_so_far = avg
        	best_clf_so_far = i
    
    for clf_name in sorted(avg_dict, key=avg_dict.get, reverse=True):
        print("{}: {}".format(clf_name, avg_dict[clf_name]))
    
    return avg_dict, clfs[best_clf_so_far]




def getCLF(finalFilePath, model_confidence_csv_path, clf_file, recalculate=True):
    if not recalculate:
#        prediction result (y_result) not available
        return joblib.load(clf_file), None
    
#    First load the data from csv file
    data = pd.read_csv(finalFilePath)
    
#    Drop columns that are not needed and normalized each columns
    data = prepare_data(data, drop_na=True)
    data = data.loc[(data['FTR'] == 'H') | (data['FTR'] == 'D') | (data['FTR'] == 'A')]
    
#   Divide data into features and label
    X_all = data.drop(columns=['FTR'])
    y_all = data['FTR']

#   List of Classifiers that we are going to run
    classifiers = [
                # Logistic Regressions
                LogisticRegression(),
                # Best param in this grid search
                LogisticRegression(penalty='l2', solver='newton-cg', multi_class='ovr',
                                   C=0.1, warm_start=True),
                LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',
                                   C=0.4, warm_start=False),
               # SVC
                SVC(probability=True),
                SVC(C=0.3, class_weight=None, decision_function_shape='ovo', degree=1,
                    kernel='rbf', probability=True, shrinking=True, tol=0.0005),
                SVC(C=0.28, class_weight=None, decision_function_shape='ovo', degree=1,
                    kernel='rbf', probability=True, shrinking=True, tol=0.0002),
                # XGBoost
                xgb.XGBClassifier(),
                xgb.XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=2,
                    min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.7,
                    scale_pos_weight=0.8, reg_alpha=1e-5, booster='gbtree', objective='multi:softprob'),
#                KNeighborsClassifier(),
#                RandomForestClassifier(),
#                GaussianNB(),
#                DecisionTreeClassifier(),
#                GradientBoostingClassifier(),
#                LinearSVC(),
#                SGDClassifier()
            ]
    
    
##    Example of how to grid search classifiers
##    Logistic Regression
#    clf_L = LogisticRegression()
#    parameters_L = {'penalty': ['l2'], 
#                    'solver': ['lbfgs', 'newton-cg', 'sag'], 
#                    'multi_class': ['ovr', 'multinomial'],
#                    'C': [x * 0.1 + 0.1 for x in range(10)],
#                    'warm_start': [True, False],
#                    'fit_intercept':[True, False],
#                    'class_weight':['balanced',None]}
#    f1_scorer_L = make_scorer(f1_score, labels=['H','D','A'], average = 'micro')
#    clf_L = get_grid_clf(clf_L, f1_scorer_L, parameters_L, X_all, y_all)
#    classifiers.append(clf_L)
    
##    SVC
#    clf_L = SVC()
#    parameters_L = {
#            'C': [x * 0.01 + 0.27 for x in range(5)], 
#            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#            'degree': [x + 1 for x in range(3)],
#            'shrinking': [True, False],
#            'tol':[x * 0.0005 + 0.0005 for x in range(3)],
#            'class_weight':['balanced',None],
#            'decision_function_shape': ['ovo', 'ovr']
#            }
#    f1_scorer_L = make_scorer(f1_score, labels=['H','D','A'], average = 'micro')
#    clf_L = get_grid_clf(clf_L, f1_scorer_L, parameters_L, X_all, y_all)
#    classifiers.append(clf_L)
    
##    XGBoost
#    clf_L = xgb.XGBClassifier()
#    parameters_L = {
#            'learning_rate': [0.01],
#            'n_estimators':[1000],
#            'max_depth': [2],
#            'min_child_weight': [5],
#            'gamma': [0],
#            'subsample': [0.8],
#            'colsample_bytree': [0.7],
#            'scale_pos_weight':[0.8],
#            'reg_alpha':[1e-5],
#            'booster': ['gbtree'],
#            'objective': ['multi:softprob']
#            }
#    f1_scorer_L = make_scorer(f1_score, labels=['H','D','A'], average = 'micro')
#    clf_L = get_grid_clf(clf_L, f1_scorer_L, parameters_L, X_all, y_all)
#    classifiers.append(clf_L)
    
#   We are going to record accuracies of each classifier prediction iteration
    len_classifiers = len(classifiers)
    result = [[] for _ in range(len_classifiers)]
    y_results = [[] for _ in range(len_classifiers + 1)]
    
#   Using 10-fold cross validation (Dividing the data into sub groups (90% to fit, 10% to test), and run 
#   prediction with each classifiers using the sub groups as a dataset)
    split = 10
    kf = KFold(n_splits=split, shuffle=True)
    for split_index, (train_index, test_index) in enumerate(kf.split(X_all)):
        print("Processing {}/{} of KFold Cross Validation...".format(split_index + 1, split))
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
        y_results[len_classifiers] += y_test.tolist()
        
        for index, clf in enumerate(classifiers):
            print("KFold: {}/{}. clf_index: {}/{}.".format(split_index + 1, split, index + 1, len(classifiers)))
            confidence, predicted_result = train_predict(clf, X_train, y_train, X_test, y_test)
            result[index].append(confidence)
            y_results[index] += predicted_result.tolist()
    
#   Make a dictionary of average accuracies for each classifiers
    avg_dict, best_clf = process_print_result(classifiers, result)
    
#   Put the result into csv file
    if os.path.isfile(model_confidence_csv_path):    
        df = pd.read_csv(model_confidence_csv_path)
        newdf = pd.DataFrame(avg_dict, index=[df.shape[1]])
        df = pd.concat([df, newdf], ignore_index=True, sort=False)
    else:
        make_directory(model_confidence_csv_path)
        df = pd.DataFrame(avg_dict, index=[0])
    df.to_csv(model_confidence_csv_path, index=False)
    
#    Saves the classifier using joblib module
    if recalculate:
        joblib.dump(best_clf, clf_file)
#   Return the best classifier
    return best_clf, y_results
```
