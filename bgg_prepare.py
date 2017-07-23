"""Stores global variables for use in this project, runs the full preparation script if executed."""

_FIRST_DATABASE = "boardgames.sqlite"
_SECOND_DATABASE = "boardgames-clean.sqlite"
_SVD_FILE = "ratings_svd.pickle"

import bgg_collect
import bgg_clean
import bgg_optimize

def fetch_clean_prepare():
	"""Runs the full preparation script."""
	bgg_collect.process_game_list()
	bgg_clean.main()
	bgg_optimize.main()

if __name__=="__main__":
	fetch_clean_prepare()