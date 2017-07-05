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

from os.path import isfile
import sqlite3

_DATABASE_NAME = "boardgames-clean.sqlite"
_DB_GAMES = None
_DB_LINKS = None
_DB_RATINGS = None
_QUERY_SCHEMA = "query_schema.json"
_DEFAULT_INPUT_QUERY = "input.json"

def _load_db(file=_DATABASE_NAME):
	"""Loads the given BGG database into memory for use in this module."""
	global _DB_GAMES, _DB_LINKS, _DB_RATINGS
	conn = sqlite3.connect(file)
	_DB_GAMES = pd.read_sql_query("SELECT * FROM games", conn)
	_DB_LINKS = pd.read_sql_query("SELECT * FROM links", conn)
	_DB_RATINGS = pd.read_sql_query("SELECT * FROM ratings", conn)
	conn.close()

def _translate_json(target):
	"""Loads the given file as a JSON object or translates the given string into a JSON object."""
	if isfile(target):
		return json.load(open(target,"rt"))
	return json.loads(target)

def get_link_types():
	"""Returns a list of all the link types in the database. Returns a pandas Series."""
	global _DB_LINKS
	return _DB_LINKS.type.unique()

def filter_links(type, filter=""):
	"""Finds all link values of the given type with a substring matching the filter. Returns a pandas Series."""
	global _DB_LINKS
	right_type = pd.Series(_DB_LINKS[_DB_LINKS.type==type].value.unique())
	return right_type[right_type.apply(unidecode).str.lower().str.find(unidecode(filter).lower())!=-1]

def _filter_by_name(target, query=None):
	"""Filters the given pandas dataframe by the "name" query given. If the query is None, returns the original dataframe."""
	if query==None:
		return target
	if "require" in query:
		target = target[target.name.apply(unidecode).str.lower()==unidecode(query["require"]).lower()]
	if "contains" in query:
		target = target[target.name.apply(unidecode).str.lower().find(unidecode(query["contains"]).lower())!=-1]
	if "regex" in query:
		target = target[target.name.apply(unidecode).str.lower().str.contains(unidecode(query["regex"]).lower())]
	return target

def _filter_by_gameId(target, query=None):
	"""Filters out all games in the provided datagrame with gameId not found in the query. If the query is None, returns the original dataframe."""
	if query==None:
		return target
	if isinstance(query,int):
		target = target[target.gameId==query]
	else:
		target = target[target.gameId.apply(lambda x: x in query)]
	return target

def _filter_by_description(target, query=None):
	"""Retains only games with the specified substring and that match the specified regex. If the query is None, returns the original dataframe."""
	if query==None:
		return target
	if "contains" in query:
		target = target[target.description.apply(unidecode).str.lower().find(unidecode(query["contains"]).lower())!=-1]
	if "regex" in query:
		target = target[target.description.apply(unidecode).str.lower().contains(unidecode(query["regex"]).lower())]
	return target

def _filter_by_image(target, query=None):
	"""Filters games by presence or absence of an image URL. If the query is None, returns the original dataframe."""
	if query==None:
		return target
	if "require" in query:
		if query["require"]:
			target = target[target.image.notnull()]
		else:
			target = target[target.image.isnull()]
	return target

def _filter_by_year(target, query=None):
	"""Filters games by their publication year. If the query is None, returns the original dataframe."""
	if query==None:
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
	if query==None:
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
	if query==None:
		return target
	if "atLeast" in query:
		target = target[(target.minPlayTime>=query["atLeast"]) | target.minPlayTime.isnull()]
	if "atMost" in query:
		target = target[(target.maxPlayTime<=query["atMost"]) | target.maxPlayTime.isnull()]
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
	if query==None:
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
	if query==None:
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
	if query==None:
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
	if query==None:
		return target
	if "minExpansions" in query:
		target = target[target.numExpansions>=query["minExpansions"]]
	if "maxExpansions" in query:
		target = target[target.numExpansions<=query["maxExpansions"]]
	return target

def _filter_by_links(target, linktype, query=None):
	"""Filters by the given link type. If the query is None, returns the original dataframe."""
	global _DB_LINKS
	if query==None | ("require" not in query):
		return target
	by_lt = _DB_LINKS[_DB_LINKS.type==linktype]
	for linkvalue in query["require"]:
		lv = unidecode(linkvalue).lower()
		gids = by_lt[by_lt.value.apply(unidecode).str.lower()==lv].gameId
		has_lv = target.gameId.apply(lambda x: x in gids)
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

def _process_query(target=_DEFAULT_INPUT_QUERY):
	"""Processes the given query - returns a pandas data frame with the results after adding a new column with relative rankings for each result."""
	global _QUERY_SCHEMA, _DB_GAMES
	query = _translate_json(target)
	schema = _translate_json(_QUERY_SCHEMA)
	validate(query, schema)
	df = _process_filter_query(_DB_GAMES, query)
	pass # TODO
