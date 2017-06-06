library(RSQLite)

# Read in the tables from the database.
db.con <- dbConnect(SQLite(), dbname="boardgames.sqlite")
games <- dbReadTable(db.con, "games") # We begin with 90809 games.
links <- dbReadTable(db.con, "links") # We begin with 770982 links.
ratings <- dbReadTable(db.con, "ratings") # We begin with 11255523 ratings.
dbDisconnect(db.con)
rm(db.con)

# Remove games published in 2017 or later.
games.filter <- split(games, games$yearPublished>2016 & !is.na(games$yearPublished))
games.removed <- games.filter[["FALSE"]]
games <- games.filter[["TRUE"]]

# Remove expansions to all games.
games.filter <- split(games, games$bgType)
games.removed <- rbind(games.removed,games.filter[["boardgameexpansion"]])
games <- games.filter[["boardgame"]]
print(cat("# Removed", nrow(games.removed), "games,", nrow(games), "games remaining."))
rm(games.filter)

# Remove links associated with the removed games.
link.filter <- split(links, sapply(links$gameId, function(x){x %in% games.removed$gameId}))
links <- link.filter[["FALSE"]]
print(cat("# Removed", nrow(link.filter[["TRUE"]]), "links,", nrow(links), "links remaining."))
rm(link.filter)

# Remove ratings associated with the removed games.
rating.filter <- split(ratings, sapply(ratings$gameId, function(x){x %in% games.removed$gameId}))
ratings <- rating.filter[["FALSE"]]
print(cat("# Removed", nrow(rating.filter[["TRUE"]]), "ratings,", nrow(ratings), "ratings remaining."))
rm(rating.filter)
rm(games.removed)

# Remove any ratings from users who only rated a single game each.
ratings.by.user <- aggregate(. ~ userId, data=ratings, length)
small.users <- ratings.by.user$userId[ratings.by.user$rating==1]
ratings.filter <- split(ratings, sapply(ratings$userId, function(x){x %in% small.users}))
ratings <- ratings.filter[["FALSE"]]
print(cat("# Removed", nrow(ratings.filter[["TRUE"]]), "ratings,", nrow(ratings), "ratings remaining."))
rm(ratings.filter)
rm(small.users)
rm(ratings.by.user)

# Assume that none of these games can play themselves.
games$minPlayers[games$minPlayers==0] <- NA # 2457 games
games$maxPlayers[games$maxPlayers==0] <- NA # 6624 games
# Assume that the minimum is below the maximum.
tempMaxPlayers <- games$maxPlayers
for (i in 1:nrow(games)) {
  if (!is.na(games[i,"minPlayers"]) & !is.na(games[i,"maxPlayers"]) & games[i,"minPlayers"]>games[i,"maxPlayers"]) {
    games$maxPlayers[i] <- games$minPlayers[i]
    games$minPlayers[i] <- tempMaxPlayers[i]
  }
}
rm(tempMaxPlayers)

# Assume that all of these games take time to play.
games$minPlayTime[games$minPlayTime==0] <- NA # 20021 games
games$maxPlayTime[games$maxPlayTime==0] <- NA # 21477 games
# Assume that the minimum is below the maximum.
tempMaxPlayTime <- games$maxPlayTime
for (i in 1:nrow(games)) {
  if (!is.na(games[i,"minPlayTime"]) & !is.na(games[i,"maxPlayTime"]) & games[i,"minPlayTime"]>games[i,"maxPlayTime"]) {
    games$maxPlayTime[i] <- games$minPlayTime[i]
    games$minPlayTime[i] <- tempMaxPlayTime[i]
  }
}
rm(tempMaxPlayTime)

# Assume none of these games make sense before age 1.
games$minAge[games$minAge==0] <- NA # 22637 games

# Game rating is a 1 to 10 scale.
games$averageRating[games$averageRating==0] <- NA # 23628 games
games$bayesAverageRating[games$bayesAverageRating==0] <- NA # 69978 games

# With no ratings, the standard deviation is meaningless.
games[is.na(games$averageRating),"stddev"] <- NA # 23628 games

# Weight (complexity rating) is a 1 to 5 scale.
games$avgWeight[games$avgWeight==0] <- NA # 49558 games

# Count the number of expansions for each game.
numexps <- sapply(games$expansionList, function(x){length(unlist(strsplit(x,",")))})
games <- data.frame(games, numExpansions=numexps)

# Remove columns which won't be considered in the analysis.
games <- games[,c("gameId","name","image","description","yearPublished","minPlayers","maxPlayers","minPlayTime","maxPlayTime","minAge","usersRated","averageRating","stddev","numWeights","avgWeight","numExpansions")]

# Save the tables to a new file.
db.con <- dbConnect(SQLite(), dbname="boardgames-clean.sqlite")
dbWriteTable(db.con, "games", games)
dbWriteTable(db.con, "links", links)
dbWriteTable(db.con, "ratings", ratings)
dbDisconnect(db.con)
