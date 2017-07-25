"""Cleans the BGG database."""

import sqlite3

try:
	import pandas as pd
except ImportError:
	import pip
	pip.main(["install","pandas"])
	import pandas as pd

import numpy as np

from bgg_prepare import _FIRST_DATABASE, _SECOND_DATABASE

def main():
	"""Executes the entire cleaning script."""
	global _FIRST_DATABASE, _SECOND_DATABASE
	
	# Read in the tables from the database.
	conn = sqlite3.connect(_FIRST_DATABASE)
	games = pd.read_sql_query("SELECT * FROM games", conn) # We begin with 90809 games.
	links = pd.read_sql_query("SELECT * FROM links", conn) # We begin with 770982 links.
	ratings = pd.read_sql_query("SELECT * FROM rawratings", conn) # We begin with 11255523 ratings.
	conn.close()
	
	# Remove games published in 2017 or later.
	games_removed = games[games.yearPublished>2016]
	games = games[(games.yearPublished<=2016) | games.yearPublished.isnull()]
	
	# Remove expansions to all games.
	games_removed = games_removed.append(games[games.bgType=="boardgameexpansion"])
	games = games[games.bgType!="boardgameexpansion"]
	print("# Removed %s games, %s games remaining."%(len(games_removed),len(games)))
	
	# Remove links associated with the removed games.
	links_removed = links[links.gameId.isin(games_removed.gameId)]
	links = links[links.gameId.isin(games.gameId)]
	print("# Removed %s links, %s links remaining."%(len(links_removed),len(links)))
	del links_removed
	
	# Remove ratings associated with the removed games.
	ratings_removed = ratings[ratings.gameId.isin(games_removed.gameId)]
	ratings = ratings[ratings.gameId.isin(games.gameId)]
	print("# Removed %s ratings, %s ratings remaining."%(len(ratings_removed),len(ratings)))
	del ratings_removed
	del games_removed
	
	# Remove any ratings from users who only rated a single game each.
	ratings_count = ratings.groupby("userId").size().rename("numRatings").reset_index()
	small_users = ratings_count.userId[ratings_count.numRatings<=1]
	ratings_removed = ratings[ratings.userId.isin(small_users)]
	ratings = ratings[~ratings.userId.isin(small_users)]
	print("# Removed %s ratings, %s ratings remaining."%(len(ratings_removed),len(ratings)))
	del ratings_count
	del small_users
	del ratings_removed
	
	# Replace absent descriptions with the empty string.
	games.loc[games.description.isnull(),"description"] = ""
	
	# Assume that none of these games can play themselves.
	games.loc[games.minPlayers==0,"minPlayers"] = np.NaN # 2457 games
	games.loc[games.maxPlayers==0,"maxPlayers"] = np.NaN # 6624 games
	# Assume that the minimum is below the maximum.
	where_swapped = games.minPlayers > games.maxPlayers
	temp_minPlayers = games.minPlayers[where_swapped]
	games.loc[where_swapped,"minPlayers"] = games.maxPlayers[where_swapped]
	games.loc[where_swapped,"maxPlayers"] = temp_minPlayers
	del where_swapped
	del temp_minPlayers
	
	# Assume that all of these games take time to play.
	games.loc[games.minPlayTime==0,"minPlayTime"] = np.NaN # 20021 games
	games.loc[games.maxPlayTime==0,"maxPlayTime"] = np.NaN # 21477 games
	# Assume that the minimum is below the maximum.
	where_swapped = games.minPlayTime > games.maxPlayTime
	temp_minPlayTime = games.minPlayTime[where_swapped]
	games.loc[where_swapped,"minPlayTime"] = games.maxPlayTime[where_swapped]
	games.loc[where_swapped,"maxPlayTime"] = temp_minPlayTime
	del where_swapped
	del temp_minPlayTime
	
	# Assume none of these games make sense before age 1.
	games.loc[games.minAge==0,"minAge"] = np.NaN # 22637 games
	
	# Game rating is a 1 to 10 scale.
	games.loc[games.averageRating==0,"averageRating"] = np.NaN # 23628 games
	games.loc[games.bayesAverageRating==0,"bayesAverageRating"] = np.NaN # 69978 games
	
	# With no ratings, the standard deviation is meaningless.
	games.loc[games.averageRating.isnull(),"stddev"] = np.NaN # 23628 games
	
	# Weight (complexity rating) is a 1 to 5 scale.
	games.loc[games.avgWeight==0,"avgWeight"] = np.NaN # 49558 games
	
	# Count the number of expansions for each game.
	numexps = games.expansionList.apply(lambda x: len(x.split(",")) if len(x)>0 else 0)
	games["numExpansions"] = numexps
	
	# Remove columns which won't be considered in the analysis.
	games = games[["gameId","name","image","description","yearPublished","minPlayers","maxPlayers","minPlayTime","maxPlayTime","minAge","usersRated","averageRating","stddev","numWeights","avgWeight","numExpansions"]]
	
	# Save the tables to a new file.
	conn = sqlite3.connect(_SECOND_DATABASE)
	games.to_sql("games", conn, index=False)
	links.to_sql("links", conn, index=False)
	ratings.to_sql("ratings", conn, index=False)
	conn.close()

if __name__=="__main__":
	main()