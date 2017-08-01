"""A variety of search and filtration functions for BGG."""

import pip

try:
	import pandas as pd
except ImportError:
	pip.main(["install","pandas"])
	import pandas as pd

try:
	from unidecode import unidecode
except ImportError:
	pip.main(["install","unidecode"])
	from unidecode import unidecode

try:
	import ujson as json
except ImportError:
	import json

try:
	from jsonschema import validate
except ImportError:
	pip.main(["install","jsonschema"])
	from jsonschema import validate

try:
	from multiset import FrozenMultiset
except ImportError:
	pip.main(["install","multiset"])
	from multiset import FrozenMultiset

try:
	import sklearn
except ImportError:
	pip.main(["install","sklearn"])
	import sklearn

try:
	from tqdm import tqdm, trange
except ImportError:
	def tqdm(i,*args,**kwds):
		"""Placeholder identity function since package `tqdm` is not available."""
		return i
	
	def trange(*args, **kwds):
		"""Placeholder identity function since package `tqdm` is not available."""
		return range(args[:3])

from os.path import isfile
import sqlite3
import numpy
import string
import sys
import pickle

# With help from https://stackoverflow.com/a/39902267
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

# With help from http://pandas.pydata.org/pandas-docs/stable/indexing.html#evaluation-order-matters
pd.set_option("mode.chained_assignment",None)

#if numpy.log2(sys.maxsize)<=32:
#	raise ImportError("Must be run on a 64-bit version of Python.")

from bgg_prepare import _SECOND_DATABASE as _DATABASE_NAME
from bgg_prepare import _SVD_FILE

_DB_GAMES = None
_DB_LINKS = None
_QUERY_SCHEMA = "query_schema.json"
_DEFAULT_INPUT_QUERY = "input.json"
_RESULT_FILENAME = "query_results.sqlite"
_DEFAULT_IMPORTANCE = { "description": 1, "image": 1, "publicationYear": 1, "players": 1, "playTime": 1, "minAge": 1, "ratings": 1, "weights": 1, "expansions": 1, "category": 1, "mechanic": 1, "family": 1, "designer": 1, "artist": 1, "publisher": 1, "implementation": 1, "myRatings": 1 }

def load_db(file=_DATABASE_NAME):
	"""Loads the given BGG database into memory for use in this module."""
	global _DB_GAMES, _DB_LINKS
	conn = sqlite3.connect(file)
	_DB_GAMES = pd.read_sql_query("SELECT * FROM games", conn)
	_DB_LINKS = pd.read_sql_query("SELECT * FROM links", conn)
	conn.close()

def _translate_json(target):
	"""Loads the given file as a JSON object or translates the given string into a JSON object."""
	if isinstance(target, str):
		if isfile(target):
			return json.load(open(target,"rt"))
		return json.loads(target)
	return target

def get_link_types():
	"""Returns a list of all the link types in the database. Returns a pandas Series."""
	global _DB_LINKS
	return list(_DB_LINKS.type.unique())

def filter_links(type, filter=""):
	"""Finds all link values of the given type with a substring matching the filter. Returns a pandas Series."""
	global _DB_LINKS
	right_type = pd.Series(_DB_LINKS[_DB_LINKS.type==type].value.unique())
	return list(right_type[right_type.apply(unidecode).str.lower().str.find(unidecode(filter).lower())!=-1])

def _filter_by_name(target, query=None):
	"""Filters the given pandas dataframe by the "name" query given. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "require" in query:
		target = target[target.name.apply(unidecode).str.lower()==unidecode(query["require"]).lower()]
	if "contains" in query:
		target = target[target.name.apply(unidecode).str.lower().str.find(unidecode(query["contains"]).lower())!=-1]
	if "regex" in query:
		target = target[target.name.apply(unidecode).str.contains(query["regex"])]
	return target

def _filter_by_gameId(target, query=None):
	"""Filters out all games in the provided datagrame with gameId not found in the query. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if isinstance(query,int):
		target = target[target.gameId==query]
	else:
		target = target[target.gameId.isin(query)]
	return target

def _filter_by_description(target, query=None):
	"""Retains only games with the specified substring and that match the specified regex. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "contains" in query:
		target = target[target.description.apply(unidecode).str.lower().str.find(unidecode(query["contains"]).lower())!=-1]
	if "regex" in query:
		target = target[target.description.apply(unidecode).str.contains(query["regex"])]
	return target

def _filter_by_image(target, query=None):
	"""Filters games by presence or absence of an image URL. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "require" in query:
		if query["require"]:
			target = target[target.image.notnull()]
		else:
			target = target[target.image.isnull()]
	return target

def _filter_by_year(target, query=None):
	"""Filters games by their publication year. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "exactly" in query:
		target = target[(target.yearPublished==query["exactly"]) | target.yearPublished.isnull()]
	if "before" in query:
		target = target[(target.yearPublished<=query["before"]) | target.yearPublished.isnull()]
	if "after" in query:
		target = target[(target.yearPublished>=query["after"]) | target.yearPublished.isnull()]
	if "includeMissing" in query:
		if query["includeMissing"]:
			target = target[target.yearPublished.isnull()]
		else:
			target = target[target.yearPublished.notnull()]
	return target

def _filter_by_players(target, query=None):
	"""Filters games by the number of players they can have. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "includes" in query:
		target = target[((target.minPlayers<=query["includes"]) | target.minPlayers.isnull()) & ((target.maxPlayers>=query["includes"]) | target.maxPlayers.isnull())]
	if "maxAtLeast" in query:
		target = target[(target.maxPlayers>=query["maxAtLeast"]) | target.maxPlayers.isnull()]
	if "minAtMost" in query:
		target = target[(target.minPlayers<=query["minAtMost"]) | target.minPlayers.isnull()]
	if "includeMinMissing" in query:
		if query["includeMinMissing"]:
			target = target[target.minPlayers.isnull()]
		else:
			target = target[target.minPlayers.notnull()]
	if "includeMaxMissing" in query:
		if query["includeMaxMissing"]:
			target = target[target.maxPlayers.isnull()]
		else:
			target = target[target.maxPlayers.notnull()]
	return target

def _filter_by_playtime(target, query=None):
	"""Filters games by the amount of time it takes to play. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "atLeast" in query:
		target = target[(target.minPlayTime>=query["atLeast"]) | (target.minPlayTime.isnull() & ((target.maxPlayTime>=query["atLeast"]) | target.maxPlayTime.isnull()))]
	if "atMost" in query:
		target = target[(target.maxPlayTime<=query["atMost"]) | (target.maxPlayTime.isnull() & ((target.minPlayTime<=query["atMost"]) | target.minPlayTime.isnull()))]
	if "includeMinMissing" in query:
		if query["includeMinMissing"]:
			target = target[target.minPlayTime.isnull()]
		else:
			target = target[target.minPlayTime.notnull()]
	if "includeMaxMissing" in query:
		if query["includeMaxMissing"]:
			target = target[target.maxPlayTime.isnull()]
		else:
			target = target[target.maxPlayTime.notnull()]
	return target

def _filter_by_minage(target, query=None):
	"""Filters games by the minimum recommended age. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "atLeast" in query:
		target = target[(target.minAge>=query["atLeast"]) | target.minAge.isnull()]
	if "atMost" in query:
		target = target[(target.minAge<=query["atMost"]) | target.minAge.isnull()]
	if "includeMissing" in query:
		if query["includeMissing"]:
			target = target[target.minAge.isnull()]
		else:
			target = target[target.minAge.notnull()]
	return target

def _filter_by_ratings(target, query=None):
	"""Filters games by their ratings. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "minRated" in query:
		target = target[target.usersRated>=query["minRated"]]
	if "maxRated" in query:
		target = target[target.usersRated<=query["maxRated"]]
	if "minRating" in query:
		target = target[(target.averageRating>=query["minRating"]) | target.averageRating.isnull()]
	if "maxRating" in query:
		target = target[(target.averageRating<=query["maxRating"]) | target.averageRating.isnull()]
	return target

def _filter_by_weights(target, query=None):
	"""Filters games by their weight. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "minWeighted" in query:
		target = target[target.numWeights>=query["minWeighted"]]
	if "maxWeighted" in query:
		target = target[target.numWeights<=query["maxWeighted"]]
	if "minWeight" in query:
		target = target[(target.avgWeight>=query["minWeight"]) | target.avgWeight.isnull()]
	if "maxWeight" in query:
		target = target[(target.avgWeight<=query["maxWeight"]) | target.avgWeight.isnull()]
	return target

def _filter_by_expansions(target, query=None):
	"""Filters games by how many expansions they have. If the query is None, returns the original dataframe."""
	if query is None:
		return target
	if "minExpansions" in query:
		target = target[target.numExpansions>=query["minExpansions"]]
	if "maxExpansions" in query:
		target = target[target.numExpansions<=query["maxExpansions"]]
	return target

def _filter_by_links(target, linktype, query=None):
	"""Filters by the given link type. If the query is None, returns the original dataframe."""
	global _DB_LINKS
	if (query is None) or ("require" not in query):
		return target
	by_lt = _DB_LINKS[_DB_LINKS.type==linktype]
	for linkvalue in query["require"]:
		lv = unidecode(linkvalue).lower()
		gids = by_lt[by_lt.value.apply(unidecode).str.lower()==lv].gameId
		has_lv = target.gameId.isin(gids)
		if query["require"][linkvalue]:
			target = target[has_lv]
		else:
			target = target[~has_lv]
	return target

def _process_filter_query(target, query):
	"""Processes the filtration portion of a query on dataframe target."""
	full = _filter_by_name(target, query.get("name"))
	full = _filter_by_gameId(full, query.get("gameId"))
	full = _filter_by_description(full, query.get("description"))
	full = _filter_by_image(full, query.get("image"))
	full = _filter_by_year(full, query.get("publicationYear"))
	full = _filter_by_players(full, query.get("players"))
	full = _filter_by_playtime(full, query.get("playTime"))
	full = _filter_by_minage(full, query.get("minAge"))
	full = _filter_by_ratings(full, query.get("ratings"))
	full = _filter_by_weights(full, query.get("weights"))
	full = _filter_by_expansions(full, query.get("expansions"))
	for lt in get_link_types():
		full = _filter_by_links(full, lt, query.get(lt))
	return full

def _doc_normalize(d):
	"""Performs various operations on a document (str) to normalize it, returning a multiset of its words."""
	return FrozenMultiset(unidecode(d).lower().translate(str.maketrans("","",string.punctuation)).split())

def _prefer_by_description(target, query=None):
	"""Weighs by the given search query (using TF-IDF). If the query is None, has no effect."""
	global _DEFAULT_IMPORTANCE
	if (query is None) or ("query" not in query):
		return
	q = _doc_normalize(query["query"])
	normed = target.description.apply(_doc_normalize)
	qt = frozenset([ k[0] for k in q.items() if k[1]>0 ])
	tfidf = pd.Series(0,target.index.values)
	for term in qt:
		query_term_weight = numpy.log2(1 + len(normed) / (1+normed.apply(lambda x: term in x).sum()))
		doc_term_weights = 1 + numpy.log2(normed.apply(lambda x: x.get(term,0.5)))
		tfidf += q[term]*query_term_weight*doc_term_weights
	wt = query.get("importance", _DEFAULT_IMPORTANCE["description"])
	target.sumWeights += wt
	max_tfidf = tfidf.max()
	target.sumPredictions += wt*(1 + 9*tfidf/max_tfidf)

def _prefer_by_image(target, query=None):
	"""Weighs by the given image preferences."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	if "prefer" in query:
		wt = query.get("importance", _DEFAULT_IMPORTANCE["image"])
		target.loc[:,"sumWeights"] += wt
		target.loc[target.image.notnull(),"sumPredictions"] += wt*(10 if query["prefer"] else 1)
		target.loc[target.image.isnull(),"sumPredictions"] += wt*(1 if query["prefer"] else 10)

def _prefer_by_year(target, query=None):
	"""Weighs by the given year preference."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["publicationYear"])
	if "prefer" in query:
		if query["prefer"]=="new":
			lookFor = target.yearPublished.max()
		elif query["prefer"]=="old":
			lookFor = target.yearPublished.min()
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-target.yearPublished.min()), abs(lookFor-target.yearPublished.max()))
		target.loc[target.yearPublished.notnull(),"sumWeights"] += wt
		target.loc[target.yearPublished.notnull(),"sumPredictions"] += wt*(10-9*(lookFor-target.yearPublished[target.yearPublished.notnull()]).abs()/denom)
	if "preferKnown" in query:
		target.loc[target.yearPublished.isnull(),"sumWeights"] += wt
		target.loc[target.yearPublished.isnull(),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)

def _prefer_by_players(target, query=None):
	"""Weighs by the given player-number preference."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["players"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = target.maxPlayers.max()
		elif query["prefer"]=="low":
			lookFor = target.minPlayers.min()
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-target.minPlayers.max()), abs(lookFor-target.maxPlayers.min()))
		target.loc[(target.minPlayers>=lookFor) | (target.maxPlayers<=lookFor) | (target.minPlayers.notnull() & target.maxPlayers.notnull()),"sumWeights"] += wt
		target.loc[(target.minPlayers<lookFor) & (target.maxPlayers>lookFor),"sumPredictions"] += wt*10
		target.loc[target.minPlayers>=lookFor,"sumPredictions"] += wt*(10-9*(target.minPlayers[target.minPlayers>=lookFor]-lookFor)/denom)
		target.loc[target.maxPlayers<=lookFor,"sumPredictions"] += wt*(10-9*(lookFor-target.maxPlayers[target.maxPlayers<=lookFor])/denom)
		if "preferKnown" in query:
			target.loc[((target.minPlayers<lookFor) & target.maxPlayers.isnull()) | ((target.maxPlayers>lookFor) & target.minPlayers.isnull()),"sumWeights"] += wt
			target.loc[((target.minPlayers<lookFor) & target.maxPlayers.isnull()) | ((target.maxPlayers>lookFor) & target.minPlayers.isnull()),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)
			target.loc[target.minPlayers.isnull() & target.maxPlayers.isnull(),"sumWeights"] += wt
			target.loc[target.minPlayers.isnull() & target.maxPlayers.isnull(),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)
	elif "preferKnown" in query:
		target.sumWeights += wt
		target.sumPredictions += wt*(10 if query["preferKnown"] else 1)
		target.loc[target.minPlayers.isnull(),"sumPredictions"] += wt*4.5*(-1 if query["preferKnown"] else 1)
		target.loc[target.maxPlayers.isnull(),"sumPredictions"] += wt*4.5*(-1 if query["preferKnown"] else 1)

def _prefer_by_playtime(target, query=None):
	"""Weighs by the given playtime preference."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["playTime"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = target.maxPlayTime.max()
		elif query["prefer"]=="low":
			lookFor = target.minPlayTime.min()
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-target.minPlayTime.max()), abs(lookFor-target.maxPlayTime.min()))
		target.loc[(target.minPlayTime>=lookFor) | (target.maxPlayTime<=lookFor) | (target.minPlayTime.notnull() & target.maxPlayTime.notnull()),"sumWeights"] += wt
		target.loc[(target.minPlayTime<lookFor) & (target.maxPlayTime>lookFor),"sumPredictions"] += wt*10
		target.loc[target.minPlayTime>=lookFor,"sumPredictions"] += wt*(10-9*(target.minPlayTime[target.minPlayTime>=lookFor]-lookFor)/denom)
		target.loc[target.maxPlayTime<=lookFor,"sumPredictions"] += wt*(10-9*(lookFor-target.maxPlayTime[target.maxPlayTime<=lookFor])/denom)
		if "preferKnown" in query:
			target.loc[((target.minPlayTime<lookFor) & target.maxPlayTime.isnull()) | ((target.maxPlayTime>lookFor) & target.minPlayTime.isnull()),"sumWeights"] += wt
			target.loc[((target.minPlayTime<lookFor) & target.maxPlayTime.isnull()) | ((target.maxPlayTime>lookFor) & target.minPlayTime.isnull()),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)
			target.loc[target.minPlayTime.isnull() & target.maxPlayTime.isnull(),"sumWeights"] += wt
			target.loc[target.minPlayTime.isnull() & target.maxPlayTime.isnull(),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)
	elif "preferKnown" in query:
		target.sumWeights += wt
		target.sumPredictions += wt*(10 if query["preferKnown"] else 1)
		target.loc[target.minPlayTime.isnull(),"sumPredictions"] += wt*4.5*(-1 if query["preferKnown"] else 1)
		target.loc[target.maxPlayTime.isnull(),"sumPredictions"] += wt*4.5*(-1 if query["preferKnown"] else 1)

def _prefer_by_minage(target, query=None):
	"""Weighs by the given minimum age preference."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["minAge"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = target.minAge.max()
		elif query["prefer"]=="low":
			lookFor = target.minAge.min()
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-target.minAge.min()), abs(lookFor-target.minAge.max()))
		target.loc[target.minAge.notnull(),"sumWeights"] += wt
		target.loc[target.minAge.notnull(),"sumPredictions"] += wt*(10-9*(lookFor-target.minAge[target.minAge.notnull()]).abs()/denom)
	if "preferKnown" in query:
		target.loc[target.minAge.isnull(),"sumWeights"] += wt
		target.loc[target.minAge.isnull(),"sumPredictions"] += wt*(1 if query["preferKnown"] else 10)

def _prefer_by_ratings(target, query=None):
	"""Weighs by the given preferences on average rating. Defaults to treating higher-rated games as more desirable."""
	global _DEFAULT_IMPORTANCE
	if (query is None) or (len(query)==0):
		query={ "prefer": "high" }
	wt = query.get("importance", _DEFAULT_IMPORTANCE["ratings"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = 10
		elif query["prefer"]=="low":
			lookFor = 1
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-1), abs(10-lookFor))
		conf = numpy.arctan(target.usersRated/30)*2/numpy.pi
		target.sumWeights += wt*conf
		target.loc[target.averageRating.notnull(),"sumPredictions"] += wt*conf[target.averageRating.notnull()]*(10-9*(lookFor-target.averageRating[target.averageRating.notnull()]).abs()/denom)
	elif "preferKnown" in query:
		target.sumWeights += wt
		target.loc[target.averageRating.isnull(),"sumPredictions"] += (1 if query["preferKnown"] else 10)
		target.loc[target.averageRating.notnull(),"sumPredictions"] += (10 if query["preferKnown"] else 1)

def _prefer_by_weight(target, query=None):
	"""Weighs by the given preferences on average weight."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["weights"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = 5
		elif query["prefer"]=="low":
			lookFor = 1
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor-1), abs(5-lookFor))
		conf = numpy.arctan(target.numWeights/30)*2/numpy.pi
		target.sumWeights += wt*conf
		target.loc[target.avgWeight.notnull(),"sumPredictions"] += wt*conf[target.avgWeight.notnull()]*(10-9*(lookFor-target.avgWeight[target.avgWeight.notnull()]).abs()/denom)
	elif "preferKnown" in query:
		target.sumWeights += wt
		target.loc[target.avgWeight.isnull(),"sumPredictions"] += (1 if query["preferKnown"] else 10)
		target.loc[target.avgWeight.notnull(),"sumPredictions"] += (10 if query["preferKnown"] else 1)

def _prefer_by_expansions(target, query=None):
	"""Weighs by the given preference on number of expansions."""
	global _DEFAULT_IMPORTANCE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["expansions"])
	if "prefer" in query:
		if query["prefer"]=="high":
			lookFor = target.numExpansions.max()
		elif query["prefer"]=="low":
			lookFor = 0
		else:
			lookFor = query["prefer"]
		denom = max(abs(lookFor), abs(lookFor-target.numExpansions.max()))
		target.sumWeights += wt
		target.sumPredictions += wt*(10-9*(lookFor-target.numExpansions).abs()/denom)

def _prefer_by_links(target, linktype, query=None):
	"""Weighs by the given preferences regarding the specified link type."""
	global _DEFAULT_IMPORTANCE, _DB_LINKS
	if (query is None) or ("prefer" not in query):
		return
	wt = query.get("totalImportance", _DEFAULT_IMPORTANCE[linktype])
	by_lt = _DB_LINKS[_DB_LINKS.type==linktype]
	not_normed = pd.Series(0,target.index.values)
	for linkvalue in query["prefer"]:
		if query["prefer"][linkvalue]==0:
			continue
		lv = unidecode(linkvalue).lower()
		gids = by_lt.gameId[by_lt.value.apply(unidecode).str.lower()==lv]
		has_lv = target.gameId.isin(gids)
		if query["prefer"][linkvalue]>0:
			not_normed[has_lv] += query["prefer"][linkvalue]
		else:
			not_normed[~has_lv] -= query["prefer"][linkvalue]
	denom = sum([ abs(v) for v in query["prefer"].values() ])
	target.sumWeights += wt
	target.sumPredictions += wt*(1+9*not_normed/denom)

def _prefer_by_myratings(target, query=None):
	"""Calculates the predicted ratings based on SVD and weighs it into the calculation. Thanks to the following articles for help with learning SVD:
	https://dl2.pushbulletusercontent.com/bL9cXpE89hjTdtv6XNCaArvxF1XHpn3w/webKDD00.pdf
	https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html"""
	global _DEFAULT_IMPORTANCE, _DB_GAMES, _SVD_FILE
	if query is None:
		return
	wt = query.get("importance", _DEFAULT_IMPORTANCE["myRatings"])
	user_zero = { int(x):query[x] for x in query if x.isdigit() }
	with open(_SVD_FILE,"rb") as f:
		columns, svd = pickle.load(f)
	uz_matrix = pd.DataFrame(index=pd.Series([0]), columns=columns, dtype="float64")
	for x in user_zero:
		if x in columns:
			uz_matrix[x][0] = user_zero[x]
	uz_avg = uz_matrix.mean(1)
	game_averages = _DB_GAMES.set_index("gameId").averageRating
	uz_matrix = uz_matrix.fillna(game_averages).subtract(uz_avg, axis=0)
	uz_predict = pd.DataFrame(svd.inverse_transform(svd.transform(uz_matrix)), index=uz_matrix.index, columns=uz_matrix.columns)
	uz_predict = uz_predict.add(uz_avg, axis=0)
	uz_predict[uz_predict<1] = 1
	uz_predict[uz_predict>10] = 10
	predictions = uz_predict.transpose().reset_index().rename(columns={"index":"gameId", 0:"user_zero"})
	target.loc[target.gameId.isin(predictions.index),"sumWeights"] += wt
	temp_df = pd.merge(target, predictions, sort=False)
	target.loc[:,"sumPredictions"] += wt*temp_df.loc[:,"user_zero"].fillna(0)

def _process_preference_query(target, query):
	"""Processes the preference portion of a query on dataframe target."""
	part = target.copy()
	part["sumPredictions"] = 0.0
	part["sumWeights"] = 0.0
	_prefer_by_description(part, query.get("description"))
	_prefer_by_image(part, query.get("image"))
	_prefer_by_year(part, query.get("publicationYear"))
	_prefer_by_players(part, query.get("players"))
	_prefer_by_playtime(part, query.get("playTime"))
	_prefer_by_minage(part, query.get("minAge"))
	_prefer_by_ratings(part, query.get("ratings"))
	_prefer_by_weight(part, query.get("weights"))
	_prefer_by_expansions(part, query.get("expansions"))
	for lt in get_link_types():
		_prefer_by_links(part, lt, query.get(lt))
	_prefer_by_myratings(part, query.get("myRatings"))
	part["prediction"] = part.sumPredictions / part.sumWeights
	del part["sumWeights"]
	del part["sumPredictions"]
	return part

def process_query(target=_DEFAULT_INPUT_QUERY):
	"""Processes the given query - returns a pandas data frame with the results after adding a new column with relative rankings for each result."""
	global _QUERY_SCHEMA, _DB_GAMES, _RESULT_FILENAME
	query = _translate_json(target)
	schema = _translate_json(_QUERY_SCHEMA)
	validate(query, schema)
	resultfile = query.get("filename",_RESULT_FILENAME)
	if _DB_GAMES is None:
		load_db()
	df = _process_filter_query(_DB_GAMES, query)
	df = _process_preference_query(df, query)
	conn = sqlite3.connect(resultfile)
	df.to_sql("results",conn,index=False)
	conn.close()
	print("Finished computing results.")

if __name__=="__main__":
	process_query()