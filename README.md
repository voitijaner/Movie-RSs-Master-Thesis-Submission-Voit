# Movie-RSs-Master-Thesis-Submission-Voit
Final code submission for the master thesis "Cultural differences in various Wikipedia language versions - An investigation using a movie recommender system"


# Python environemnts 
1) For data transformations, the RDF2Vec implementations (https://github.com/IBCNServices/pyRDF2Vec/tree/4c534b4e1e6df8e6c2f0ae18a43283d5ce320858. Version from Mar 29, 2020) and the similar movie extraction a python 3.7 eviroment is required. This includes the following files:
- side_data_to_movie_list.py
- similar_movie_extractions.py
- statistic_calculations.py

2) The movie GraphLab Create recommender system needs a python 2.7 environemt. Additional requirements can be found in https://turi.com/download/install-graphlab-create-command-line.html. This environment is used for the following files:
- item_similarity_recommender_genre_performance.py
- item_similarity_recommender_number_of_recommendations_productionCountry_genres.py
- item_similarity_recommender_overall_performance.py
- item_similarity_recommender_year_performance.py


# Intital data which is needed to run the program
```bash
├───data
│   ├───cosine_similarities
│   ├───lang_versions # https://wiki.dbpedia.org/downloads-2016-10 (Mappingbased Objects and Instance Types Transitive for each langauge version, .ttl format)
│   │   └───mapping_to_other_languages 	# https://wiki.dbpedia.org/downloads-2016-10 (Interlanguage Links for the english language version, .ttl format)
│   ├───movielense	# https://grouplens.org/datasets/movielens/1m/ (all files, without folder)
│   │   └───movielense_mapping	# https://github.com/sisinflab/LODrecsys-datasets/blob/master/Movielens1M/MappingMovielens2DBpedia-1.2.tsv (renamed to movielensmapping.csv)
│   └───similar_movies
│       └───genre
```
data
├── movielense
│		https://grouplens.org/datasets/movielens/1m/ (all files, without folder)	
│   ├─ movielense_mapping
│		https://github.com/sisinflab/LODrecsys-datasets/blob/master/Movielens1M/MappingMovielens2DBpedia-1.2.tsv (renamed to movielensmapping.csv)
├─  lang_versions
│		https://wiki.dbpedia.org/downloads-2016-10 (Mappingbased Objects and Instance Types Transitive for each langauge version, .ttl format)
	├─ mapping_to_other_languages
	https://wiki.dbpedia.org/downloads-2016-10 (Interlanguage Links for the english language version, .ttl format)


# File definition
Each file contains its own method definitions or program parts, even if they are duplicated in other files. With this techinique so its easier to review and understand the code. 

1) similar_movie_extractions.py
Creates with the use of RDF2Vec python library mentioned above the embeddings for the movies and stores the similar movie lists.
2) side_data_to_movie_list.py
Transforms the MovieLense movie side data and enriches it with production country informations from Wikidata. 
3) statistic_calculations.py
Calculates the sparsity of the data sets at different points of data preprocessing. 
4) item_similarity_recommender_overall_performance.py
Calculates the overall performance over the whole data set for each language version with different k nearest neighbours
5) item_similarity_recommender_number_of_recommendations_productionCountry_genres.py
Calculates the number of recommended production countries and genres per language version over the whole dataset
6) item_similarity_recommender_genre_performance.py
Calculates the genre specific performances for each language version.
7) item_similarity_recommender_year_performance.py
Calculates the year specific performance for each langauge verison.
