import os
import json
import rdflib
import bz2
import pandas as pd
import _pickle as pickle

from SPARQLWrapper import SPARQLWrapper, JSON


def loadFinalMovieList():
    """Returns the final movie List, extracted from the language_mapping_list of the similarMovieExtraction file."""
    file = open('data/lang_versions/mapping_to_other_languages/lang_mapping_list_2016.pkl', 'rb')
    movie_mapping_list = pickle.load(file)
    return movie_mapping_list['english_version'].unique()

def loadMovielenseMovieData():
    """Returns the movie-data provided by MovieLense"""
    return pd.read_csv("data/movielense/movies.dat", sep="::", names=['movieId', 'title', 'genres'])

def loadMovieLenseMapping():
	"""Returns the movielensemapping.csv data set"""
	return pd.read_csv('data/movielense/movielense_mapping/movielensmapping.csv', sep='\t', encoding='utf-8', usecols=[0,1,2], names=['id', 'name', 'dbpediaLink'])

def loadEnglishToOtherLanguages():
	"""Returns the interlanguage_links_en.tll file and returns it with removed brackets."""
	interlanguage_links_en = pd.read_csv('data/lang_versions/mapping_to_other_languages/interlanguage_links_en.ttl', sep='\s', usecols=[0,1,2], names=['subject', 'predicate', 'object'], header=1, encoding="utf-8")
	return remove_brackets(interlanguage_links_en)

def remove_brackets(df):
	"""Remove starting and ending brackets for subject, predicate and object of the given RDF-Dataset."""
	df['subject'] = df['subject'].str.replace('<', '')
	df['subject'] = df['subject'].str.replace('>', '')
	df['predicate'] = df['predicate'].str.replace('>', '')
	df['predicate'] = df['predicate'].str.replace('<', '')
	df['object'] = df['object'].str.replace('>', '')
	df['object'] = df['object'].str.replace('<', '')
	return df

def get_country_Movie_list(final_movies):
    """Returns a list which contains the the countries for the final movies"""
    #Filter wikidata for query
    get_country_Movie_list = loadEnglishToOtherLanguages()
    interlanguage_links_en_filtered = interlanguage_links_en[interlanguage_links_en['object'].str.contains("http://www.wikidata.org/entity/")]
    #filters the wikidata entries with the movies from the final movie list
    interlanguage_links_en_filtered = interlanguage_links_en_filtered[interlanguage_links_en_filtered['subject'].isin(final_movies['movie_name'].unique())] 
    count = 0
    result_list = pd.DataFrame()
    #multiple queries are necessary, because of limits / restrictions for wikidata sparql queries.
    for i in range(0,2000,500):
        query_rows = interlanguage_links_en_filtered[i: i+500]
        query_params = ""
        for i, r in query_rows.iterrows():
            #extracts the Wikidata movie identifier and creates the query parameter
            query_params = query_params + "wd:" + r['object'].replace("http://www.wikidata.org/entity/", "") + " "
        result_from_row = get_country_from_wikidata(query_params)
        result_list = result_list.append(result_from_row, ignore_index = True)
    result_list = pd.merge(result_list, interlanguage_links_en_filtered, left_on='item', right_on='object', how="left")
    result_list = result_list.drop(['object', 'predicate', 'item'], axis=1)

    return(result_list)

def get_country_from_wikidata(query_params):
    """This function queries the Wikidata spqrql with the given query parameter - movie list."""
    #Creates the final query. wdt:495 = Country of origin parametr in Wikidata
    query = """
    SELECT 
        ?item ?countryLabel 
    WHERE {
        VALUES ?item { """ +query_params+ """}
        ?item wdt:P495 ?country .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
     }
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query) 
    result = sparql.query()
    #The following lines transform the query results
    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']
    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)
    return pd.DataFrame(out, columns=cols)

def store_movie_list_genre_year_county(movie_list):
    """Store the final final movie list with genre, year and country properties for the movie RSs."""
    with open("data/movielense/final_movie_genre_year_county_list.pkl", 'wb') as f:
        pickle.dump(movie_list, f)
    movie_list.to_csv("data/movielense/final_movie_genre_year_county_list.csv")


def getTransformendMovielenseDataListWithGenreYear(final_movies):
    """Loads and transforms the movie side data provided by MovieLense and returnes a 
    movie list with genre and year for the final movies."""
    movie_side_data = loadMovielenseMovieData()
    movielenseMapping = loadMovieLenseMapping()

    #transform the genres and extract the year from the title
    for i, row in movie_side_data.iterrows():
        movie_side_data.at[i,'genres'] = movie_side_data.loc[i].genres.split("|")
        movie_side_data.at[i,'year'] = movie_side_data.loc[i]['title'][-5:-1]

    #Add infos to the final movie list
    final_movies['genres'] = ""
    final_movies['year'] = ""
    final_movies['movieId'] = ""
    for i, row in final_movies.iterrows():
        id = movielenseMapping[movielenseMapping['dbpediaLink'] == row['movie_name']]['id'].values[0]
        movie_side_data_per_movie = movie_side_data[movie_side_data['movieId'] == id]
        final_movies.at[i, 'genres'] = movie_side_data_per_movie['genres'].values[0]
        final_movies.at[i, 'movieId'] = id
        final_movies.at[i, 'year'] = movie_side_data_per_movie['year'].values[0]
    
    #Reshape the list in final form
    movie_list_genre_year = final_movies['genres'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('genres').join(final_movies[['movie_name', 'movieId', 'year']], how='left')
    return movie_list_genre_year


# load final movie list
final_movies = loadFinalMovieList()
final_movies = pd.DataFrame(final_movies, columns=['movie_name'])

# get genre and year for movies in final movies
movie_list_genre_year = getTransformendMovielenseDataListWithGenreYear(final_movies)

# get countries for final movies
country_movie_list = get_country_Movie_list(final_movies)

# merge genres / year list and countries list
movie_list_genre_year_country = pd.merge(movie_list_genre_year, country_movie_list, left_on='movie_name', right_on='subject', how="left")

movie_list_genre_year_country = movie_list_genre_year_country.drop(['subject'], axis=1)
movie_list_genre_year_country = movie_list_genre_year_country.rename(columns={'countryLabel': 'Country'})

# store the resulting list
store_movie_list_genre_year_county(movie_list_genre_year_country)