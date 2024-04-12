import pandas as pd

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import leagueleaders
from nba_api.stats.endpoints import boxscoreplayertrackv2 
from nba_api.stats.endpoints import draftcombinespotshooting 
from nba_api.stats.endpoints import draftcombinenonstationaryshooting 
from nba_api.stats.endpoints import draftcombinestats 
from nba_api.stats.endpoints import draftcombineplayeranthro 
from nba_api.stats.endpoints import playercareerstats 
from nba_api.stats.static import players
#from nba_api.stats.endpoints import playercareerstats

player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544)
#, proxy='127.0.0.1:80', headers=custom_headers, timeout=100)

df = player_info.get_data_frames()

#career = playercareerstats.PlayerCareerStats(player_id='203999')
#df = career.get_data_frames()

players = players.find_players_by_last_name('^(james|love)$')

# Pull data for the top 500 scorers
top_500 = leagueleaders.LeagueLeaders(
	season='2023-24',
	season_type_all_star='Regular Season',
	stat_category_abbreviation='PTS'
	,per_mode48='PerGame'
).get_data_frames()[0][:500]

# Correct column names for grouping
avg_stats_columns = ['PLAYER', 'PLAYER_ID', 'MIN', 'FGM', 'FGA', 'FTM', 'FTA', 'PTS', 'FG3M', 'FG3A']
#top_500_avg = top_500.groupby(['PLAYER', 'PLAYER_ID'])[avg_stats_columns].mean()
top_500_avg = top_500[avg_stats_columns].sort_values(by='PTS',ascending=False)

# Inspect the first few rows of the averaged stats
print(top_500_avg.head())

y = boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id='0021700807')

year = [2021, 2022, 2023]

spotShooting = draftcombinespotshooting.DraftCombineSpotShooting(season_year=2021).get_data_frames()[0]
nonStationaryShooting = draftcombinenonstationaryshooting.DraftCombineNonStationaryShooting(season_year=2021).get_data_frames()[0]
measurements = draftcombinestats.DraftCombineStats(season_all_time=2021).get_data_frames()[0]

spotShootingTrunc = spotShooting[['PLAYER_ID', 'COLLEGE_CORNER_LEFT_MADE', 'COLLEGE_CORNER_LEFT_ATTEMPT', 'COLLEGE_CORNER_LEFT_PCT']][ \
(spotShooting['COLLEGE_CORNER_LEFT_MADE'].notnull()) \
].sort_values(by='PLAYER_ID', ascending=False)

nonStationaryShootingTrunc = nonStationaryShooting[['PLAYER_ID', 'OFF_DRIB_COLLEGE_BREAK_LEFT_MADE', 'OFF_DRIB_COLLEGE_BREAK_LEFT_ATTEMPT', 'OFF_DRIB_COLLEGE_BREAK_LEFT_PCT'\
, 'ON_MOVE_COLLEGE_MADE', 'ON_MOVE_COLLEGE_ATTEMPT', 'ON_MOVE_COLLEGE_PCT']][ \
(nonStationaryShooting['OFF_DRIB_COLLEGE_BREAK_LEFT_MADE'].notnull() | nonStationaryShooting['ON_MOVE_COLLEGE_MADE'].notnull()) \
].sort_values(by='PLAYER_ID', ascending=False)

draftPlayers = measurements[['PLAYER_ID', 'FIRST_NAME', 'LAST_NAME', 'HEIGHT_WO_SHOES', 'HEIGHT_W_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH',\
'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'MODIFIED_LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT']\
].sort_values(by='PLAYER_ID', ascending=False)

df = pd.merge(draftPlayers, spotShootingTrunc, on='PLAYER_ID', how='outer')
df = pd.merge(df, nonStationaryShootingTrunc, on='PLAYER_ID', how='outer')
dfFiltered = df[~(df['COLLEGE_CORNER_LEFT_ATTEMPT'].isnull() & df['OFF_DRIB_COLLEGE_BREAK_LEFT_ATTEMPT'].isnull() & df['ON_MOVE_COLLEGE_ATTEMPT'].isnull())]

colsPct = ['PLAYER_ID', 'FIRST_NAME', 'LAST_NAME', 'HEIGHT_W_SHOES', 'WINGSPAN', 'MAX_VERTICAL_LEAP', 'COLLEGE_CORNER_LEFT_PCT', 'OFF_DRIB_COLLEGE_BREAK_LEFT_PCT', 'ON_MOVE_COLLEGE_PCT']
colMeasurement = ['PLAYER_ID', 'FIRST_NAME', 'LAST_NAME', 'HEIGHT_WO_SHOES', 'HEIGHT_W_SHOES', 'WEIGHT', 'WINGSPAN', 'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH',\
'STANDING_VERTICAL_LEAP', 'MAX_VERTICAL_LEAP', 'LANE_AGILITY_TIME', 'MODIFIED_LANE_AGILITY_TIME', 'THREE_QUARTER_SPRINT']

print(dfFiltered[colsPct].sort_values(by='HEIGHT_W_SHOES', ascending=False))

draft2021 = dfFiltered['PLAYER_ID']

import pdb; pdb.set_trace()
