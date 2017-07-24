"""Tests the prediction power of the SVD algorithm and optimizes its various parameters for these circumstances. Thanks to the following websites for aid:
https://github.com/neilsummers/predict_movie_ratings/blob/master/movieratings.py
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html"""

import sqlite3
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from os.path import isfile
from os import remove
import pickle

from bgg_prepare import _SECOND_DATABASE as _DB_NAME
from bgg_prepare import _SVD_FILE

_GAME_AVG = None
_SAVE_FILE = "bgg_optimize.pickle"
_TEST_DB_NAME = "bgg-optimize.sqlite"
_STOP_FILE = "stop.command"

def load_db():
	"""Loads the game averages from the database."""
	global _GAME_AVG, _DB_NAME
	conn = sqlite3.connect(_DB_NAME)
	_GAME_AVG = pd.read_sql_query("select gameId, averageRating from games where averageRating not null", conn).set_index("gameId").averageRating
	conn.close()

def make_rating_matrix(ratings, min_count, rating_counts=None, dtype="float64"):
	"""Creates the normalized and filled rating matrix for all users who have rated at least `min_count` games."""
	global _GAME_AVG
	if rating_counts is None:
		rating_counts = pd.DataFrame(ratings.groupby("userId").size().rename("num_ratings")).reset_index()
	uids = pd.Index(rating_counts.userId[rating_counts.num_ratings>=min_count])
	local_ratings = ratings[ratings.userId.isin(uids)]
	gids = pd.Index(local_ratings.gameId.unique())
	rating_matrix = pd.DataFrame(index=uids, columns=gids, dtype=dtype)
	for row in tqdm(local_ratings.itertuples(), desc="Making Rating Matrix", leave=False, total=len(local_ratings), miniters=1):
		rating_matrix[row[2]][row[1]] = row[3]
	user_averages = rating_matrix.mean(1)
	rating_matrix = rating_matrix.fillna(_GAME_AVG)
	rating_matrix = rating_matrix.subtract(user_averages, axis=0)
	return rating_matrix

def make_test_matrix(ratings_q, ratings_a, gids, dtype="float64"):
	"""Translates a ratings table pair for testing into a test-relevant matrix. Returns a pair consisting of 1) The matrix to be translated as the test, and 2) The matrix considered correct answers."""
	global _GAME_AVG
	uids = pd.Index(ratings_q.userId.unique())
	local_ratings_q = ratings_q[ratings_q.gameId.isin(gids)]
	local_ratings_a = ratings_a[ratings_a.gameId.isin(gids)]
	rat_mat_q = pd.DataFrame(index=uids, columns=gids, dtype=dtype)
	rat_mat_a = pd.DataFrame(index=uids, columns=gids, dtype=dtype)
	for row in tqdm(local_ratings_q.itertuples(), desc="Making Test Matrix [1/2]", leave=False, total=len(local_ratings_q), miniters=1):
		rat_mat_q[row[2]][row[1]] = row[3]
	for row in tqdm(local_ratings_a.itertuples(), desc="Making Test Matrix [2/2]", leave=False, total=len(local_ratings_a), miniters=1):
		rat_mat_a[row[2]][row[1]] = row[3]
	user_averages = rat_mat_q.mean(1)
	rat_mat_q = rat_mat_q.fillna(_GAME_AVG)
	rat_mat_q = rat_mat_q.subtract(user_averages, axis=0)
	rat_mat_a = rat_mat_a.subtract(user_averages, axis=0)
	return [rat_mat_q, rat_mat_a]

def make_svd(rating_matrix, n_components, random_seed, n_iter=5):
	"""Creates an SVD transformer from the given rating matrix."""
	svd = TruncatedSVD(n_components=n_components, random_state=random_seed, n_iter=n_iter)
	svd.fit(rating_matrix)
	return svd

def _setup_database(filename, train, test_q, test_a):
	"""Sets up the storage database for results."""
	conn = sqlite3.connect(filename)
	train.to_sql("TrainRatings", conn, index=False)
	test_q.to_sql("TestRatingsQ", conn, index=False)
	test_a.to_sql("TestRatingsA", conn, index=False)
	conn.execute("""CREATE TABLE Results (
		minCount INTEGER,
		nComponents INTEGER,
		numUsers INTEGER,
		numGames INTEGER,
		seed INTEGER,
		error REAL
		)""")
	conn.commit()
	conn.close()

def calculate_varieties(savefile=_SAVE_FILE, filename=_TEST_DB_NAME, min_count=[1,5501,1], n_components=[1,1001,1], random_seed=1):
	"""Calculates the various possibilities for the relevant parameters "min_count" and "n_components", calculates the mean square error in each case, and returns True if it completes successfully."""
	global _STOP_FILE
	if isfile(savefile):
		with open(savefile,"rb") as f:
			filename, min_count, n_components, start_at, random_seed = pickle.load(f)
	else:
		start_at=n_components[0]
	conn = sqlite3.connect(filename)
	train_ratings = pd.read_sql_query("SELECT * FROM TrainRatings", conn)
	test_ratings_q = pd.read_sql_query("SELECT * FROM TestRatingsQ", conn)
	test_ratings_a = pd.read_sql_query("SELECT * FROM TestRatingsA", conn)
	rating_counts = pd.DataFrame(train_ratings.groupby("userId").size().rename("num_ratings")).reset_index()
	for mc in tqdm(range(*min_count), desc="Min Count", miniters=1):
		with open(savefile,"wb") as f:
			pickle.dump([filename, [mc,min_count[1],min_count[2]], n_components, start_at, random_seed],f)
		if isfile(_STOP_FILE):
			conn.close()
			return False
		rat_mat = make_rating_matrix(train_ratings, mc, rating_counts)
		num_u = len(rat_mat.index)
		num_g = len(rat_mat.columns)
		test_mat, answer_mat = make_test_matrix(test_ratings_q, test_ratings_a, rat_mat.columns)
		mask = answer_mat.notnull()
		for nc in tqdm(range(start_at,min(n_components[1],num_u,num_g),n_components[2]), desc="Num Comps", leave=False, miniters=1):
			with open(savefile,"wb") as f:
				pickle.dump([filename, [mc,min_count[1],min_count[2]],n_components, nc, random_seed],f)
			if isfile(_STOP_FILE):
				conn.close()
				return False
			svd = make_svd(rat_mat, nc, random_seed)
			q_mat = pd.DataFrame(svd.inverse_transform(svd.transform(test_mat)), index=test_mat.index, columns=test_mat.columns)
			error = mse(q_mat[mask].fillna(0), answer_mat.fillna(0))
			conn.execute("INSERT INTO Results VALUES (?,?,?,?,?,?)", [mc,nc,num_u,num_g,random_seed,error])
			conn.commit()
			random_seed += 1
		start_at = n_components[0]
	if isfile(savefile):
		remove(savefile)
	conn.close()
	return True

def initialize_testing(ratings, filename=_TEST_DB_NAME, min_test_ratings=40, min_train_ratings=100, test_samples=1000):
	"""Divides up ratings data into training and testing groups, with the testing group separated into question and answer tables. Also creates the database and stores these tables in it."""
	print("Counting ratings.")
	rating_counts = pd.DataFrame(ratings.groupby("userId").size().rename("num_ratings")).reset_index()
	print("Splitting ratings.")
	train_uids = rating_counts.userId[rating_counts.num_ratings>=min_train_ratings]
	test_uids = rating_counts.userId[(rating_counts.num_ratings<min_train_ratings) & (rating_counts.num_ratings>=min_test_ratings)]
	test_uids = resample(test_uids, replace=False, n_samples=test_samples, random_state=1217)
	train = ratings[ratings.userId.isin(train_uids)]
	test = ratings[ratings.userId.isin(test_uids)]
	test_q, test_a = train_test_split(test, test_size=0.25, random_state=738, stratify=test.userId)
	print("Setting up database.")
	_setup_database(filename, train, test_q, test_a)

def _get_min_error(filename=_TEST_DB_NAME):
	"""Recovers the parameters associated with the minimum error seen so far."""
	conn = sqlite3.connect(filename)
	min_row = conn.execute("SELECT minCount, nComponents, min(error) FROM Results")
	conn.close()
	return { "minCount": min_row[0], "nComponents": min_row[1] }

def main():
	"""Executes the optimization routine."""
	global _DB_NAME, _GAME_AVG, _TEST_DB_NAME, _SVD_FILE, _STOP_FILE
	load_db()
	if not isfile(_TEST_DB_NAME):
		conn = sqlite3.connect(_DB_NAME)
		print("Reading ratings.")
		ratings = pd.read_sql_query("SELECT * FROM ratings", conn)
		ratings = ratings[ratings.gameId.isin(_GAME_AVG.index)]
		conn.close()
		initialize_testing(ratings)
	if not calculate_varieties(min_count=[100,1501,100], n_components=[100,1001,100], random_seed=15):
		print("Stopped early.")
		remove(_STOP_FILE)
		return
	best = _get_min_error()
	if not calculate_varieties(min_count=[max(100,best["minCount"]-100),best["minCount"]+101,10], n_components=[max(10,best["nComponents"]-100),best["nComponents"]+101,10], random_seed=197):
		print("Stopped early.")
		remove(_STOP_FILE)
		return
	best = _get_min_error()
	if not calculate_varieties(min_count=[max(100,best["minCount"]-10),best["minCount"]+11,1], n_components=[max(1,best["nComponents"]-10),best["nComponents"]+11,1], random_seed=32954):
		print("Stopped early.")
		remove(_STOP_FILE)
		return
	best = _get_min_error()
	conn = sqlite3.connect(_TEST_DB_NAME)
	ratings = pd.read_sql_query("SELECT * FROM TrainRatings", conn)
	conn.close()
	rating_matrix = make_rating_matrix(ratings, best["minCount"])
	svd = make_svd(rating_matrix, best["nComponents"], 992)
	with open(_SVD_FILE,"wb") as f:
		pickle.dump([rating_matrix.columns, svd],f)
	print("Done optimizing.")

if __name__=="__main__":
	main()