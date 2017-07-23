"""Scrapes all available board game data from https://www.boardgamegeek.com ."""

import sys
ver = sys.version_info
if (ver[0]==2 and ver<(2,7,9)) or (ver[0]==3 and ver<(3,4)) or ver[0] not in [2,3]:
	raise ImportError("Version of python is too low; please run with version 2.7.9+ or 3.4+.")

try:
	import requests
except ImportError:
	import pip
	pip.main(["install","requests"])
	import requests

try:
	from tqdm import tqdm, trange
except ImportError:
	def tqdm(i,*args,**kwds):
		"""Placeholder identity function since package `tqdm` is not available."""
		return i
	
	def trange(*args, **kwds):
		"""Placeholder identity function since package `tqdm` is not available."""
		return range(args[:3])

import sqlite3
from re import findall, finditer
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from os.path import exists
from os import remove
from pickle import load, dump
from time import sleep
from math import ceil

try:
	import ujson as json
except ImportError:
	import json

from bgg_prepare import _FIRST_DATABASE as _DATABASE_NAME

_CHUNK_SIZE = 100
_SAVE_FILE = "bgg_collect_state.pickle"
_TEMP_XML_SAVE = "temp.xml"
_DEFAULT_CURRENCY = "CAD" # Must be a valid currency symbol.
_CONDITION_RANKINGS = [ "new", "likenew", "verygood", "good", "acceptable" ]
_BGG_SITEMAP = "https://www.boardgamegeek.com/sitemapindex"
_BGG_GAMELIST_PAGE = "http://boardgamegeek.com/sitemap_geekitems_boardgame_page_\\d+"
_BGG_GAMENUM_RE = "<loc>http://boardgamegeek.com/boardgame/(\\d+)/[^<]*</loc>"
_BGG_XMLAPI_PAGE = "https://www.boardgamegeek.com/xmlapi2/thing"
_CURRENCY_RATE_PAGE = "http://api.fixer.io/latest"
_CONDITION_FILE = "condition_rankings.txt"
_FETCH_FREQUENCY = timedelta(seconds=5)
_last_fetch = None

def _getpage(page, params=None):
	"""Fetches a page, at most once per 5 seconds."""
	global _FETCH_FREQUENCY, _last_fetch
	now = datetime.utcnow()
	if _last_fetch != None:
		diff = (_FETCH_FREQUENCY - (now-_last_fetch)).total_seconds()
		if diff>0:
			sleep(diff)
	_last_fetch = datetime.utcnow()
	return requests.get(page, params=params)

def retrieve_game_numbers():
	"""Finds the numbers of all board games on BGG via their site map. Adapted from https://github.com/9thcirclegames/bgg-analysis ."""
	global _BGG_SITEMAP, _BGG_GAMELIST_PAGE, _BGG_GAMENUM_RE
	pages = findall(_BGG_GAMELIST_PAGE,_getpage(_BGG_SITEMAP).text)
	nums = []
	for listpage in tqdm(pages,desc="Nums",leave=False):
		nums.extend([ m.group(1) for m in finditer(_BGG_GAMENUM_RE,_getpage(listpage).text) ])
	return nums

def fetch_page(nums, commentspage=None):
	"""Retrieves information on all the enumerated games. Returns a `request`-type object."""
	global _BGG_XMLAPI_PAGE
	params = { "id": ",".join(nums) }
	if commentspage!=None:
		params.update({ "ratingcomments": "1", "page": str(commentspage), "pagesize": "100" })
	if commentspage==None or commentspage==1:
		params.update({ "stats": "1", "marketplace": "1" })
	r = _getpage(_BGG_XMLAPI_PAGE,params=params)
	while r.content==b"<html><body><h1>504 Gateway Time-out</h1>\nThe server didn't respond in time.\n</body></html>\n" or len(r.content)==0 or r.content==b"<?xml version=\"1.0\" encoding=\"utf-8\"?>\t<div class='messagebox error'>\n\t\terror reading chunk of file\n\t</div>":
		r = _getpage(_BGG_XMLAPI_PAGE,params=params)
	return r

def _process_language_dependence(ld):
	"""Extract, aggregate, and return the language dependence score of a game."""
	if ld==None:
		return None
	res = 0
	tot = _int(ld.attrib["totalvotes"])
	if tot==0:
		return None
	answers = ld.findall("./results/result")
	if len(answers)<5:
		return None
	for i in range(5):
		res += (i+1)*_int(answers[i].attrib["numvotes"])
	return res / tot

def _int(x):
	"""Exactly the same as the built-in `int` function, except it returns `None` when input is `None` instead of raising an error."""
	return None if x==None or x=="" else int(x)

def _float(x):
	"""Exactly the same as the built-in `float` function, except it returns `None` when input is `None` instead of raising an error."""
	return None if x==None or x=="" else float(x)

def _find_attrib(item, xpath, attribute=None):
	"""Performs `item.find(xpath).attrib[attribute]`, but if it returns `None` at any stage, returns `None`. If `attribute=None`, returns `item.find(xpath).text` instead."""
	if item==None:
		return None
	tmp = item.find(xpath)
	if tmp==None:
		return None
	if attribute==None:
		return tmp.text
	return tmp.attrib[attribute]

def process_page(page, conn, save_temp = False, first_page=True):
	"""Processes the result object from a single page. `conn` is a connection object to the database which stores this data."""
	global _TEMP_XML_SAVE
	if save_temp:
		with open(_TEMP_XML_SAVE,"wb") as f:
			f.write(page.content)
	try:
		root = ET.fromstring(page.text.replace("\x10","").replace("\x00","").replace("\x05","").replace("\x03","").replace("\x08","").replace("\x02","").replace("\x1f","").replace("\x1d",""))
	except:
		return False
	fetch_time = datetime.utcnow().timestamp()
	item_list = root.findall("./item")
	if len(item_list)==0:
		return False
	c = conn.cursor()
	for i in item_list:
		gameId = int(i.attrib["id"])
		rating_list = []
		for r in i.findall("./comments/comment"):
			rating_list.append((gameId,r.attrib["username"],_float(r.attrib["rating"])))
		if first_page:
			pretup = [gameId]
			pretup.append(i.attrib["type"]) # BGType
			pretup.append(_find_attrib(i,"./name[@type='primary']","value")) # Name
			pretup.append(_find_attrib(i,"./image",None)) # Image
			pretup.append(_find_attrib(i,"./description",None)) # Description
			pretup.append(_int(_find_attrib(i,"./yearpublished","value"))) # YearPublished
			pretup.append(_int(_find_attrib(i,"./minplayers","value"))) # MinPlayers
			pretup.append(_int(_find_attrib(i,"./maxplayers","value"))) # MaxPlayers
			pretup.append(_int(_find_attrib(i,"./minplaytime","value"))) # MinPlayTime
			pretup.append(_int(_find_attrib(i,"./maxplaytime","value"))) # MaxPlayTime
			pretup.append(_int(_find_attrib(i,"./minage","value"))) # MinAge
			pretup.append(_process_language_dependence(i.find("./poll[@name='language_dependence']"))) # LanguageDependence
			pretup.append(_int(_find_attrib(i,"./poll[@name='language_dependence']","totalvotes"))) # LanguageDependenceVotes
			pretup.append(_int(_find_attrib(i,"./statistics/ratings/usersrated","value"))) # UsersRated
			pretup.append(_float(_find_attrib(i,"./statistics/ratings/average","value"))) # AverageRating
			pretup.append(_float(_find_attrib(i,"./statistics/ratings/bayesaverage","value"))) # BayesAverageRating
			pretup.append(_float(_find_attrib(i,"./statistics/ratings/stddev","value"))) # Stddev
			pretup.append(_int(_find_attrib(i,"./statistics/ratings/numweights","value"))) # NumWeights
			pretup.append(_float(_find_attrib(i,"./statistics/ratings/averageweight","value"))) # AvgWeight
			pretup.append(_int(_find_attrib(i,"./comments","totalitems"))) # Numratings
			pretup.append(fetch_time) # Updated
			c.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,?)",tuple(pretup))
			link_list = []
			for link in i.findall("./link"):
				link_list.append((gameId,_int(link.attrib["id"]),link.attrib["type"],link.attrib["value"]))
			if len(link_list)>0:
				c.executemany("INSERT INTO rawlinks VALUES (?,?,?,?)",link_list)
			listinglist = []
			for listing in i.findall("./marketplacelistings/listing"):
				listinglist.append((gameId,_float(_find_attrib(listing,"./price","value")),_find_attrib(listing,"./price","currency"),_find_attrib(listing,"./condition","value")))
			if len(listinglist)>0:
				c.executemany("INSERT INTO market VALUES (?,?,?,?)",listinglist)
		if len(rating_list)>0:
			c.executemany("INSERT INTO rawratings VALUES (?,?,?)",rating_list)
	conn.commit()
	c.close()
	return True

def setup_database(filename=_DATABASE_NAME):
	"""Sets up an empty SQLite database for storage and retrieval, if the given file does not exist. Returns a connection to the file."""
	if exists(filename):
		conn = sqlite3.connect(filename)
		conn.execute("pragma foreign_keys=on")
		conn.commit()
		return conn
	conn = sqlite3.connect(filename)
	conn.execute("pragma foreign_keys=on")
	c=conn.execute("""CREATE TABLE games (
		gameId INTEGER PRIMARY KEY,
		bgType TEXT,
		name TEXT,
		image TEXT,
		description TEXT,
		yearPublished INTEGER,
		minPlayers INTEGER,
		maxPlayers INTEGER,
		minPlayTime INTEGER,
		maxPlayTime INTEGER,
		minAge INTEGER,
		languageDependence REAL,
		languageDependenceVotes INTEGER,
		usersRated INTEGER,
		averageRating REAL,
		bayesAverageRating REAL,
		stddev REAL,
		numWeights INTEGER,
		avgWeight REAL,
		numRatings INTEGER,
		newPrice REAL,
		damageDepreciation REAL,
		priceSpread REAL,
		numPrices INTEGER,
		expansionList TEXT,
		updated REAL
		)""")
	c.execute("""CREATE TABLE rawlinks (
		gameId INTEGER NOT NULL,
		linkId INTEGER NOT NULL,
		type TEXT,
		value TEXT,
		FOREIGN KEY (gameId) REFERENCES games(gameId)
		)""")
	c.execute("""CREATE TABLE market (
		gameId INTEGER NOT NULL,
		price REAL,
		currency TEXT,
		condition TEXT,
		FOREIGN KEY (gameId) REFERENCES games(gameId)
		)""")
	c.execute("""CREATE TABLE rawratings (
		gameId INTEGER NOT NULL,
		username TEXT NOT NULL,
		rating REAL NOT NULL,
		FOREIGN KEY (gameId) REFERENCES games(gameId)
		)""")
	conn.commit()
	c.close()
	return conn

def _chunker(l, n):
	"""Splits a list into chunks of size at most `n`. Adapted from http://stackoverflow.com/a/434321 ."""
	return [ l[i:i+n] for i in range(0,len(l),n) ]

def _flatten(l):
	"""Flattens a list of 1-length tuples."""
	return [ k[0] for k in l ]

def _calculate_prices(price_info, rates, conditions):
	"""Calculates the price indicators (`newprice`, `damagedepreciation`, and `pricespread`) from the given price information."""
	by_condition = dict([ (k,[]) for k in conditions.keys() ])
	for t in price_info:
		by_condition[t[2]].append(t[0]*rates[t[1]])
	supp = [ k for k in by_condition.keys() if len(by_condition[k])>0 ]
	if len(supp)==0:
		return (None, None, None)
	n = len(price_info)
	if len(supp)==1: # Assume a horizontal line at the mean of the y-values.
		cond = supp[0]
		newprice,slope,damagedepreciation = sum(by_condition[cond])/len(by_condition[cond]),0,conditions[cond]
	else: # Perform linear regression
		sx = sum([ conditions[a]*len(by_condition[a]) for a in by_condition.keys() ])
		sy = sum([ sum(by_condition[a]) for a in by_condition.keys() ])
		sxs = sum([ (conditions[a]**2)*len(by_condition[a]) for a in by_condition.keys() ])
		sxy = sum([ conditions[a]*b for a in by_condition.keys() for b in by_condition[a] ])
		slope = (n*sxy-sx*sy)/(n*sxs-sx*sx)
		newprice = (sy-sx*slope)/n
		damagedepreciation = slope/newprice
	price_residuals = 0
	for k in by_condition.keys():
		predicted = 1+damagedepreciation*conditions[k]
		price_residuals += sum([ abs((p/newprice)-predicted) for p in by_condition[k] ])
	pricespread = price_residuals/n
	return (newprice, damagedepreciation, pricespread, len(price_info))

def _fix_publishing_year(conn, save_original_data=False):
	"""Fixes the publication year - there is no year 0, so it is a placeholder."""
	conn.execute("UPDATE games SET yearPublished=NULL WHERE yearPublished=0")
	conn.commit()

def _grab_currencies(conn, save_original_data=False):
	"""Retrieves and stores the currency conversion rates at present, with thanks to http://fixer.io for conversion rates."""
	global _CURRENCY_RATE_PAGE, _DEFAULT_CURRENCY
	c = conn.execute("SELECT DISTINCT currency FROM market")
	currencies = _flatten(c.fetchall())
	r = _getpage(_CURRENCY_RATE_PAGE, params={"base": _DEFAULT_CURRENCY, "symbols": ",".join(currencies) })
	rates = json.loads(r.text)["rates"]
	dates.update({ _DEFAULT_CURRENCY: 1 })
	c.execute("CREATE TABLE currencies ( currency TEXT PRIMARY KEY, exchange REAL, base INTEGER CHECK ( base IN (0,1) ) )")
	c.executemany("INSERT INTO currencies VALUES (?,?,?)",[ (a,b,(1 if b==1 else 0)) for a,b in rates.items() ])
	conn.commit()

def _setup_condition_file(conn=None, save_original_data=False):
	"""Creates a condition-ranking file, if it doesn't already exist."""
	global _CONDITION_FILE, _CONDITION_RANKINGS
	if not exists(_CONDITION_FILE):
		with open(_CONDITION_FILE,"wt") as f:
			f.write("### This file should consist of the various conditions in which a game can be found, ranked in ascending order of damage. Do not alter this line.")
			for cond in _CONDITION_RANKINGS:
				f.write("\n"+cond)

def _store_condition_rankings(conn, save_original_data=False):
	"""Stores the ranked conditions in the database."""
	global _CONDITION_FILE
	with open(_CONDITION_FILE,"rt") as f:
		condition_rankings = [ rank.strip() for rank in f.readlines()[1:] ]
	c = conn.execute("CREATE TABLE conditionrankings ( condition TEXT PRIMARY KEY, rank INTEGER )")
	k = 0
	conds = _flatten(c.execute("SELECT DISTINCT condition FROM market").fetchall()) + ["new"]
	missing_conditions = set(conds)-set(condition_rankings)
	if len(missing_conditions)!=0:
		raise KeyError(missing_conditions)
	for q in condition_rankings:
		if q in conds:
			c.execute("INSERT INTO conditionrankings VALUES (?,?)",(q,k))
			k += 1
	conn.commit()

def _calculate_all_prices(conn, save_original_data=False):
	"""Calculates all the game prices through linear regression, and cleans up temporary tables if necessary."""
	c = conn.cursor()
	rates = dict(c.execute("SELECT currency, exchange FROM currencies"))
	conds = dict(c.execute("SELECT * FROM conditionrankings").fetchall())
	for a in tqdm(c.execute("SELECT DISTINCT gameId FROM market").fetchall(),desc="Market",leave=False):
		x=_calculate_prices(c.execute("SELECT price, currency, condition FROM market WHERE gameId=?",a).fetchall(), rates, conds)+(a[0],)
		c.execute("UPDATE games SET newPrice=?, damageDepreciation=?, priceSpread=?, numPrices=? WHERE gameId=?",x)
	c.execute("UPDATE games SET numPrices=0 WHERE numPrices IS NULL")
	if not save_original_data:
		c.execute("DROP TABLE conditionRankings")
		c.execute("DROP TABLE market")
		c.execute("DROP TABLE currencies")
	conn.commit()

def _fix_ratings(conn, save_original_data=False):
	"""Deletes any user who only rated one game (as they will be useless for collaborative filtering) and averages the ratings of any user who has rated the same game multiple times."""
	c = conn.execute("CREATE TABLE ratecount AS SELECT username, COUNT(DISTINCT gameId) AS ngid FROM rawratings GROUP BY username")
	c.execute("CREATE TABLE tempratings AS SELECT gameId, rawratings.username as username, rating FROM rawratings, ratecount WHERE rawratings.username=ratecount.username AND ratecount.ngid!=1")
	conn.execute("CREATE TABLE ratings AS SELECT username, gameId, AVG(rating) AS rating FROM tempratings GROUP BY username, gameId")
	c.execute("DROP TABLE ratecount")
	c.execute("DROP TABLE tempratings")
	if not save_original_data:
		c.execute("DROP TABLE rawratings")
	conn.commit()

def _collect_expansions(conn, save_original_data=False):
	"""Removes links of type `boardgameexpansion` from the `links` table, collecting them into the `expansionList` column of the `games` table."""
	c = conn.execute("SELECT gameId, linkId FROM rawlinks WHERE type='boardgameexpansion'")
	exps = c.fetchall()
	ids = _flatten(c.execute("SELECT DISTINCT gameId FROM rawlinks WHERE type='boardgameexpansion'").fetchall())
	c.execute("CREATE TABLE links AS SELECT * FROM rawlinks WHERE type!='boardgameexpansion'")
	exp_lists = dict([ (i,[]) for i in ids ])
	for a,b in exps:
		exp_lists[a].append(b)
	c.executemany("UPDATE games SET expansionList=? WHERE gameId=?",[ (",".join([ str(b) for b in sorted(exp_lists[a]) ]),a) for a in exp_lists ])
	if not save_original_data:
		c.execute("DROP TABLE rawlinks")
	conn.commit()

def _anonymize_users(conn, save_original_data=False):
	"""Replaces all usernames with arbitrary userIds, not connected to their BGG userId."""
	c = conn.execute("ALTER TABLE ratings RENAME TO namedratings")
	c.execute("CREATE TABLE usernumbers (userId INTEGER PRIMARY KEY, username TEXT)")
	c.execute("INSERT INTO usernumbers(username) SELECT DISTINCT username FROM namedratings")
	c.execute("CREATE TABLE ratings AS SELECT usernumbers.userId AS userId, gameId, rating FROM usernumbers, namedratings WHERE usernumbers.username=namedratings.username")
	if not save_original_data:
		c.execute("DROP TABLE namedratings")
		c.execute("DTOP TABLE usernumbers")
	conn.commit()

def _prettify_link_types(conn, save_original_data=False):
	"""Removes the prefix `boardgame` from each link type."""
	conn.execute("UPDATE links SET type = substr(type,10)")
	conn.commit()

# This is a list of pre-processing functions to be performed after the initial scraping for data.
_PRE_PROCESS = [None, None, _fix_publishing_year, _grab_currencies, _setup_condition_file, _store_condition_rankings, _calculate_all_prices, _fix_ratings, _collect_expansions, _anonymize_users, _prettify_link_types]

def _cleanup(filename):
	"""Executes the `VACUUM` command on the given sqlite database. Must only be executed when the database is not in use."""
	conn = sqlite3.connect(filename, isolation_level=None)
	conn.execute("VACUUM")
	conn.close()

def process_game_list(savefile=_SAVE_FILE, filename=_DATABASE_NAME, chunk_size=_CHUNK_SIZE, save_each_page = False, save_original_data=False):
	"""Fetches and processes the entire list of games into the given database. Will save state before each chunk is processed and resume if a previous saved state exists at the start. Returns whether or not the operation completed."""
	global _TEMP_XML_SAVE, _PRE_PROCESS
	if exists(savefile):
		with open(savefile,"rb") as f:
			chunk_list, start_at, filename, stage, start_at2 = load(f)
	else:
		chunk_list = _chunker(retrieve_game_numbers(), chunk_size)
		start_at=0
		stage = 0
		start_at2 = 0
	conn = setup_database(filename)
	if stage==0:
		for k in trange(start_at,len(chunk_list),initial=start_at,total=len(chunk_list),desc="Fetching"):
			with open(savefile,"wb") as f:
				dump([chunk_list, k, filename, stage, 0], f)
			if not process_page(fetch_page(chunk_list[k],1),conn,save_temp=save_each_page):
				conn.close()
				return False
		stage = 1
		start_at=1
	if stage==1:
		max_pages = ceil(conn.execute("SELECT MAX(numratings) FROM games").fetchall()[0][0]/100)
		for k in trange(start_at,max_pages,initial=start_at,total=max_pages,desc="Ratings"):
			chunk_list = _chunker(_flatten(conn.execute("SELECT gameId FROM games WHERE numratings>=?",(k*100,)).fetchall()),chunk_size)
			for k2 in trange(start_at2,len(chunk_list),initial=start_at2,total=len(chunk_list),desc="Page #%s"%(k+1),leave=False):
				with open(savefile,"wb") as f:
					dump([chunk_list, k, filename, stage, k2], f)
				if not process_page(fetch_page([ str(gi) for gi in chunk_list[k2] ],k+1),conn,save_temp=save_each_page,first_page=False):
					conn.close()
					return False
			start_at2=0
		stage=2
	if exists(_TEMP_XML_SAVE):
		remove(_TEMP_XML_SAVE)
	oldstage=stage
	for stage in range(oldstage, len(_PRE_PROCESS)):
		with open(savefile,"wb") as f:
			dump([None,0,filename,stage,0], f)
		_PRE_PROCESS[stage](conn, save_original_data)
	if exists(savefile):
		remove(savefile)
	conn.close()
	_cleanup(filename)
	return True

if __name__=="__main__":
	process_game_list()