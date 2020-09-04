import os
import pandas as pd
import _pickle as pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from rdf2vec.converters import rdflib_to_kg
from rdf2vec.converters import create_kg
from rdf2vec.walkers import RandomWalker
from rdf2vec import RDF2VecTransformer





def loadInstanceTypes(lang):
	"""Returns the instance type for the specified language version with removed brackets."""
	instance_types = pd.read_csv('data/lang_versions/instance_types_transitive_'+lang+'.ttl', sep='\s', usecols=[0,1,2], names=['subject', 'predicate', 'object'], header = 1, encoding="utf-8")
	#last line in file contains no data
	instance_types = instance_types[:-1]
	return remove_brackets(instance_types)

def loadMappingBasedObjects(lang):
	"""Returns the mapping based objects type for the specified language version with removed brackets."""
	mappingbased_objects = pd.read_csv('data/lang_versions/mappingbased_objects_'+lang+'.ttl', sep='\s', usecols=[0,1,2], names=['subject', 'predicate', 'object'], header=1, encoding = "utf-8")
	#last line in file contains no data
	mappingbased_objects = mappingbased_objects[:-1]
	return remove_brackets(mappingbased_objects)

def loadEnglishToOtherLanguages():
	"""Returns the interlanguage_links_en.tll file and returns it with removed brackets."""
	interlanguage_links_en = pd.read_csv('data/lang_versions/mapping_to_other_languages/interlanguage_links_en.ttl', sep='\s', usecols=[0,1,2], names=['subject', 'predicate', 'object'], header=1, encoding = "utf-8")
	return remove_brackets(interlanguage_links_en)

def loadMovieLenseMapping():
	"""Returns the movielensemapping.csv dataset"""
	return pd.read_csv('data/movielense/movielense_mapping/movielensmapping.csv', sep='\t', encoding='utf-8', usecols=[0,1,2], names=['id', 'name', 'dbpediaLink'])

def remove_brackets(df):
	"""Remove starting and ending brackets for subject, predicate and object of the given RDF-Dataset."""
	df['subject'] = df['subject'].str.replace('<', '')
	df['subject'] = df['subject'].str.replace('>', '')
	df['predicate'] = df['predicate'].str.replace('>', '')
	df['predicate'] = df['predicate'].str.replace('<', '')
	df['object'] = df['object'].str.replace('>', '')
	df['object'] = df['object'].str.replace('<', '')
	return df

def extractMovies(instance_types):
	"""Extract movie entities from the instance-types dataset."""
	entities = instance_types[instance_types['object']=="http://schema.org/Movie"]['subject']
	return entities

def getMovieListFiltered():
	"""Returns a movie list with DBpedia-links, contained in the MovieLense mapping data set."""
	#Loads the english instance typies for movie extractions.
	instance_types = loadInstanceTypes("en")
	entities = extractMovies(instance_types)
	movielenseMapping = loadMovieLenseMapping()
	entities_filtered = entities[entities.isin(movielenseMapping['dbpediaLink'])]
	return entities_filtered

def getTransformer(depth, walks_perGraph, sg, vector_size):
	"""Creates and returns the transformer used by the RDF2Vec algorithm."""
	random_walker = RandomWalker(depth, walks_perGraph)
	transformer = RDF2VecTransformer(walkers=[random_walker], sg=sg, vector_size=vector_size)
	return transformer

def getEmbeddings(transformer, kg, entities):
	"""Creates and returns embeddings for the entities from the given knowledge graph, transformer."""
	return transformer.fit_transform(kg, entities)

def getCosineSimDf(embeddings, entities):
    """For the given embeddings and entities list, the cosine similarity matrix is calculated and returend. """
    cosine_sim = cosine_similarity(embeddings)
    return pd.DataFrame(data=cosine_sim, index=entities, columns=entities)

def getNHighestRated(n, movie, cosine_df):
	"""Extract the n most similar movies for the one given movie from the cosine similarity df."""
	extracted_row = cosine_df[movie]
	return extracted_row.nlargest(n+1)[1:]

def get_top_N_Items_with_Names(item_list, n, cosine_df, movielenseMapping):
	"""For each item in the item_list, the top n similar movies are extracted from the cosine_df. 
	The movielenseMapping is to enrich the similar movies with the movieId.
	
	Not recently used in program."""
	result = pd.DataFrame(columns=['movieId', 'title', 'similar_movies'])
	for item in item_list:
		n_items = getNHighestRated(n, item, cosine_df)
		n_item_list = n_items.index.tolist()
		row_result = {'movieId': movielenseMapping[movielenseMapping['dbpediaLink'] == item].iloc[0]['id'], 'title': item.replace('http://dbpedia.org/resource/', ''), 'similar_movies': [w.replace('http://dbpedia.org/resource/', '') for w in n_item_list]}
		result = result.append(row_result, ignore_index = True)
	return result

def get_k_nearest_items_with_score(item_list, n, cosine_df):
    """For each item in the item_list, the top n similar movies are extracted and returend with the consine score from the cosine_df."""
    movielenseMapping = loadMovieLenseMapping()
    result = pd.DataFrame(columns=['movie_id', 'similar', 'score', 'movie_name', 'similar_name'])

    for item in item_list:
        n_items = getNHighestRated(n, item, cosine_df)
        movie_names_list = n_items.index.tolist()
        score_list = n_items.values
        for i in range(n):
            movie_id_item = movielenseMapping[movielenseMapping['dbpediaLink'] == item].iloc[0]['id']
            movie_id_simiilar = movielenseMapping[movielenseMapping['dbpediaLink'] == movie_names_list[i]].iloc[0]['id']
            row_result = {'movie_id': movie_id_item , 'similar': movie_id_simiilar, 'score': score_list[i], 'movie_name': item, 'similar_name': movie_names_list[i]}
            result = result.append(row_result, ignore_index = True)
    return result


def create_and_store_k_nearest_genre_items_with_score(item_list, k, cosine_df, lang):
    """For each item in the item_list, the top n similar movies per genre are extracted 
    and stored with the consine score from the cosine_df. """
    counter = 0

    movielenseMapping = loadMovieLenseMapping()

    genre_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movie_name', 'genres'])
    result = pd.DataFrame(columns=['movie_id', 'similar', 'score', 'movie_name', 'similar_name'])
    genre_movie_list = genre_movie_list.drop_duplicates()
    final_genre_list = []

    #extract genres
    for genre in genre_movie_list['genres'].unique():
        genre_movies = genre_movie_list[genre_movie_list['genres'] == genre]['movie_name'].values
        if len(genre_movies) >= 100:
            final_genre_list.append([genre, genre_movies])

    # filter cosine similarity matrix and create the similar movie results per genre. 
    for genre in final_genre_list:
        cosine_df_filtered = cosine_df.filter(items=genre[1]).filter(genre[1], axis=0)

        for item in genre[1]:
            n_items = getNHighestRated(k, item, cosine_df_filtered)
            movie_names_list = n_items.index.tolist()
            score_list = n_items.values
            for i in range(k):
                movie_id_item = movielenseMapping[movielenseMapping['dbpediaLink'] == item].iloc[0]['id']
                movie_id_simiilar = movielenseMapping[movielenseMapping['dbpediaLink'] == movie_names_list[i]].iloc[0]['id']
                row_result = {'movie_id': movie_id_item , 'similar': movie_id_simiilar, 'score': score_list[i], 'movie_name': item, 'similar_name': movie_names_list[i]}
                result = result.append(row_result, ignore_index = True)
        store_k_nearest_items_genre_lang(lang,k,result,genre[0])

def store_k_nearest_items_genre_lang(lang,k,k_nearest_list, genre):
    """Stores the similar items for the movie RSs. The language, genre and k are used for identification."""
    with open("data/similar_movies/genre/"+str(k)+"_nearest_items_"+genre+"_lang="+lang+".pkl", 'wb') as f:
        pickle.dump(k_nearest_list, f)
    k_nearest_list.to_csv("data/similar_movies/genre/"+str(k)+"_nearest_items_"+genre+"_lang="+lang+".csv")

def store_k_nearest_items_lang(lang,k,k_nearest_list):
    """Stores the similar items for the movie RSs. The language and and k are used for identification."""
    with open("data/similar_movies/"+str(k)+"_nearest_items_lang="+lang+".pkl", 'wb') as f:
        pickle.dump(k_nearest_list, f)
    k_nearest_list.to_csv("data/similar_movies/"+str(k)+"_nearest_items_lang="+lang+".csv")

def store_cosine_sim_df(lang,cosine_df):
	"""Stores the cosine_df for later use. The language is used for identification."""
	with open("data/cosine_similarities/cosine_df_lang="+lang+".pkl", 'wb') as f:
		pickle.dump(cosine_df, f)


def createCosineSim(lang_list, lang_mapping_list):
	"""For the given language list, the n = 50 similar movies lists for the movie RSs are created and stored.
	The lang_mapping_list is used for mapping english DBpedia links to the other DBpedia language versions."""
	for l in lang_list:
		print("Start Language: " + l)
		#The English DBpedia version has no "en" in it, therefore the entities have to be extracted in another way than for other languages.
		if(l == "en"):
			entities_from_lang_list = lang_mapping_list[lang_mapping_list.other_version.str.contains("http://dbpedia.org/")]
		else:
			entities_from_lang_list = lang_mapping_list[lang_mapping_list.other_version.str.contains("http://"+l+".dbpedia.org/")]
		#Loads the instance types and the mapping based objects for current the language version in one list
		all_data = pd.concat([loadInstanceTypes(l), loadMappingBasedObjects(l)])
		print("Data Length" + str(len(all_data)))
		#Creates the KG
		kg = create_kg(all_data.itertuples(index=False), label_predicates=[])
		del all_data
		#Get the DBpedia links for current language
		entites_in_lang = entities_from_lang_list['other_version']
		entites_filtered = entites_in_lang[entites_in_lang.isin(entites_in_lang)]
		transformer = getTransformer(4, 500, 5, 200)
		embeddings = getEmbeddings(transformer, kg, entites_in_lang)
		del kg
		del transformer
		#Get english DBpedia link
		entites_filtered_english = entities_from_lang_list[entities_from_lang_list['other_version'].isin(entites_filtered)]['english_version']
		#Create the cosine similarity matrix
		cosine_sim = getCosineSimDf(embeddings, entites_filtered_english)
		#Store the cosine similarity matrix for current language
		store_cosine_sim_df(l, cosine_sim)
		#Extract the k most similar movies.
		k_nearest_items = get_k_nearest_items_with_score(entites_filtered_english, 50, cosine_sim)
		store_k_nearest_items_lang(l, 50, k_nearest_items)
		create_and_store_k_nearest_genre_items_with_score(entites_filtered_english, 10, cosine_sim, l)

def storeLangMappingList(list):
	"""Stores the DBpedia language mapping list."""
	with open('data/lang_versions/mapping_to_other_languages/lang_mapping_list_2016.pkl', 'wb') as f:
		pickle.dump(list, f)

def getMoviesForOtherLanguages(entities, dbpedia_versions, interlanguage_list_en):
    """Extracts for the entities and given DBpedia versions the english movie links and the corresponding 
	link in for the other langauge versions of DBpedia"""
    interlanguage_list_en_filtered = interlanguage_list_en[interlanguage_list_en.subject.isin(list(entities))]
    result = pd.DataFrame(columns=['english_version', 'other_version'])
    for entitie in entities:
		#Extracts the entries from the english interlanguage links data set
        same_as_df = interlanguage_list_en_filtered.loc[interlanguage_list_en_filtered['subject'] == entitie]
		#For each dbpedia version it is checked, if there is an corresponding entrie for the current entitie
        for link in db_pedia_versions['links']:
            sameAs = same_as_df[same_as_df.object.str.contains(link)]
            if sameAs.empty != True:
                row_result = {'english_version':entitie, 'other_version':sameAs['object'].values[0]}
                result = result.append(row_result, ignore_index = True)
    # only movies wich are in all provided dbpedia language version are stored
    result = result.groupby('english_version').filter(lambda x: len(x) == len(dbpedia_versions))
    storeLangMappingList(result)
    return result


#Load the movie list contained in the english dbpedia version and movielense mapping.
entities_filtered_en = getMovieListFiltered()
interlanguage_links = loadEnglishToOtherLanguages()
db_pedia_versions = pd.DataFrame(["http://de.dbpedia.org/", "http://dbpedia.org/", "http://fr.dbpedia.org/", "http://it.dbpedia.org/", "http://ru.dbpedia.org/"],columns=['links'])
lang_mapping_list = getMoviesForOtherLanguages(entities_filtered_en, db_pedia_versions, interlanguage_links)


lang_list = ["en", "de", "fr", "it", "ru"]

createCosineSim(lang_list, lang_mapping_list)