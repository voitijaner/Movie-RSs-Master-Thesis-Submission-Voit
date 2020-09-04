import pandas as pd


"""The genre, year and production country specific genres can be found in the corresponding files."""

def loadMovieLenseMapping():
	"""Returns the movielensemapping.csv dataset"""
	return pd.read_csv('data/movielense/movielense_mapping/movielensmapping.csv', sep='\t', encoding='utf-8', usecols=[0,1,2], names=['id', 'name', 'dbpediaLink'])

def loadFinalMovieList():
    """Returns the final movie List, extracted from the language_mapping_list of the similarMovieExtraction file."""
    file = open('data/lang_versions/mapping_to_other_languages/lang_mapping_list_2016.pkl', 'rb')
    movie_mapping_list = pickle.load(file)
    return movie_mapping_list['english_version'].unique()

def loadMovieLenseRatings():
    """Returns MovieLense ratings dataset."""
    return pd.read_csv('data/movielense/ratings.dat', sep='::', usecols=[0,1,2], names=['user_id', 'movie_id', 'rating'], encoding="utf-8")

def calculateAndPrintSparsity():
    """Calculate and prints sparsity."""

    #Load Data
    ratings = loadMovieLenseRatings()
    movielenseMapping = loadMovieLenseMapping()
    movies = loadFinalMovieList()
    #Sparsity initial data set
    sparsity = 1 - len(ratings) / (len(ratings['user_id'].unique()) * len(ratings['movie_id'].unique()))
    print("Initial")
    print("Users: " + str(len(ratings['user_id'].unique())))
    print("Movies: " + str(len(ratings['movie_id'].unique())))
    print("Sparsity: " + str(sparsity))


    #Sparsity after movie selection
    movielenseMapping = movielenseMapping[movielenseMapping.dbpediaLink.isin(movies)]

    ratings = ratings[ratings.movie_id.isin(movielenseMapping['id'])]

    sparsity = 1 - len(ratings) / (len(ratings['user_id'].unique()) * len(ratings['movie_id'].unique()))

    print("After movie selection")
    print("Users: " + str(len(ratings['user_id'].unique())))
    print("Movies: " + str(len(ratings['movie_id'].unique())))
    print("Sparsity: " + str(sparsity))


    #Sparsity after preprocessing
    ratings['frequency'] = ratings['movie_id'].map(ratings['movie_id'].value_counts())

    ratings = ratings.sort_values(by=['frequency'], ascending=False)

    count = 0
    for item in ratings['movie_id'].unique():
        if count < len(ratings['movie_id'].unique())/100:
            ratings = ratings[ratings['movie_id'] != item]
        count += 1  
    ratings = ratings.groupby('user_id').filter(lambda x: len(x) >= 50)
    ratings = ratings.drop(['frequency'], axis=1)

    sparsity = 1 - len(ratings) / (len(ratings['user_id'].unique()) * len(ratings['movie_id'].unique()))

    print("After preprocessing")
    print("Users: " + str(len(ratings['user_id'].unique())))
    print("Movies: " + str(len(ratings['movie_id'].unique())))
    print("Sparsity: " + str(sparsity))


    movies_not_in_ratings = movielenseMapping[~movielenseMapping.id.isin(ratings['movie_id'])]
    print("Movies not in ratings")
    print(movies_not_in_ratings['id'])


calculateAndPrintSparsity()
