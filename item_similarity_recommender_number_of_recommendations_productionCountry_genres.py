from __future__ import division
import graphlab as gl
import pandas as pd
from sklearn import cross_validation
import numpy as np
from collections import Counter



pd.set_option('display.max_columns', None)


# load ratings file
actions_df = pd.read_csv('data/movielense/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], encoding="utf-8")
#country movie list is already here, because it is also used for ratings filtering
country_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movieId', 'Country'])
country_movie_list = country_movie_list.drop_duplicates()
country_movie_list = country_movie_list.rename(columns={'movieId' : 'movie_id'})


# ----------- PREPROCESSING -----------

# Remove movies which are not in the final movie list
print("Initial ratings: " + str(len(actions_df)))
movie_list = country_movie_list['movie_id'].tolist()
boolean_series = actions_df.movieId.isin(movie_list)
actions_df = actions_df[boolean_series]
actions_df['frequency'] = actions_df['movieId'].map(actions_df['movieId'].value_counts())

print("Ratings after movie selection: " + str(len(actions_df)))
actions_df = actions_df.sort_values(by=['frequency'], ascending=False)

count = 0
# remove popularity bias
for item in actions_df['movieId'].unique():
    if count < len(actions_df['movieId'].unique())/100:
        actions_df = actions_df[actions_df['movieId'] != item]
    count += 1  
    
# sparcity reduction
actions_df = actions_df.groupby('userId').filter(lambda x: len(x) >= 50)
actions_df = actions_df.drop(['frequency'], axis=1)
actions_df = actions_df.rename(columns={'userId':'user_id', 'movieId':'item_id'})

print("Final ratings after preprocessing: " + str(len(actions_df)))



# transform movies to country / county-movie-list pairs
result_country_movie_list = pd.DataFrame(columns=["movie_id", "country"])
for country in country_movie_list['Country'].unique():
    movies = list(country_movie_list[country_movie_list['Country'] == country]['movie_id'])
    result_row = {'movie_id': movies, 'country': country}
    result_country_movie_list = result_country_movie_list.append(result_row, ignore_index=True)

result_country_movie_list

#country counter
Counter(result_country_movie_list['country'])


# load genre - movie list
genre_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movieId', 'genres'])
genre_movie_list = genre_movie_list.drop_duplicates()
genre_movie_list = genre_movie_list.rename(columns={'movieId' : 'movie_id'})

#transform genre list to genre / genre-movie-list pairs
final_genre_list = []
for genre in genre_movie_list['genres'].unique():
    genre_movies = genre_movie_list[genre_movie_list['genres'] == genre]['movie_id'].values
    final_genre_list.append([genre, genre_movies])


# genre statistics
for row in final_genre_list:
    print(row[0])
    temp = actions_df[actions_df['item_id'].isin(row[1])].groupby('user_id').filter(lambda x: len(x) >= 50)
    print("items: " + str(len(row[1])))
    print("ratings: " + str(len(temp['item_id'])))
    print("users: " + str(len(temp['user_id'].unique())))


# ----------- CREATE RS MODELS -----------

def create_RS(lang, k, training_data, nearest_items_sf_genre):
    """Create the item_similarity_recommender models and returns it."""
    model = gl.item_similarity_recommender.create(training_data, user_id='user_id', similarity_type="cosine", item_id='item_id', target="rating", only_top_k = k, nearest_items=nearest_items_sf_genre)
    return model



model_list = []

# create the models for each language version
for lang in ["de", "fr", "it", "ru", "en"]:
    # load similar movies
    nearest_items_df = pd.read_csv("data/similar_movies/50_nearest_items_lang="+lang+".csv", usecols=['movie_id','similar','score'])
    
    # similar movie_id and the similar movies are seen as float, convert to int
    nearest_items_df['similar'] = nearest_items_df['similar'].astype(int)
    nearest_items_df['movie_id'] = nearest_items_df['movie_id'].astype(int)
    nearest_items_df = nearest_items_df.rename(columns={"movie_id": "item_id"})
 
    nearest_items = gl.SFrame(nearest_items_df)
    training_data = gl.SFrame(actions_df)
    
    # set k to the best per language
    if lang == "de":
        k = 3
    if lang == "fr":
        k = 3
    if lang == "it":
        k = 2
    if lang == "ru":
        k = 6
    if lang == "en":
        k = 2

    model = create_RS(lang, k , training_data, nearest_items)
    model_list.append([model, lang, k])


# convert model_list into DataFrame for later use
model_list_df = pd.DataFrame(model_list, columns = ["model", "lang", "k"])
model_list_df


# ----------- EVALUATE RS MODELS -----------

# extract and store the recommended movies per language versions
predicted_movies = pd.DataFrame(columns=['recommended_movies', 'k', 'lang'])
count = 0
for i, row in model_list_df.iterrows():
    count += 1
    print(count)
    lang = row['lang']
    k = row['k']
    model = row['model']
    recommended_movies = model.recommend()
    # add recommendes movies list to predited movies list
    result_row = {'recommended_movies':recommended_movies, 'k':k, 'lang':lang}
    predicted_movies = predicted_movies.append(result_row, ignore_index=True)



#---------- START PRODUCTION COUNTRY CALCULATIONS ----------
# extract unique countries for analyse
unique_countries = country_movie_list['Country'].unique()
unique_countries = np.append(unique_countries, ['lang', 'k'])
# create result list
predicted_counts = pd.DataFrame(columns=unique_countries)

# extract recommended movies per model and add it to country list
for i, row in predicted_movies.iterrows():
    recommendations = row['recommended_movies'].to_dataframe()
    #join country movie list with recommended movies
    recommendations_with_country = pd.merge(recommendations, country_movie_list, left_on='item_id', right_on='movie_id', how="left")
    # count the movies per production country and add to result list
    row_result = recommendations_with_country['Country'].value_counts().to_dict()
    row_result['k'] = row['k']
    row_result['lang'] = row['lang']
    predicted_counts = predicted_counts.append(row_result, ignore_index=True)
            
# fill null values
predicted_counts = predicted_counts.fillna(0)
print(predicted_counts)


# sums associated movies per model
sum_rows = predicted_counts.sum(axis='columns')

print(sum_rows)
#extract unique countries
unique_countries = country_movie_list['Country'].unique()
for country in unique_countries:
    for i, row in predicted_counts.iterrows():
        #calcualte fraction
        predicted_counts.at[i, country] = predicted_counts.iloc[i][country] / (sum_rows[i] /100)

print(predicted_counts)

# used for country statistics
actions_df_country = actions_df.copy()

# calculate number of ratings per country, first join ratings list with country - movie list
actions_df_country = pd.merge(actions_df_country, country_movie_list, left_on='item_id', right_on='movie_id', how="left")
count_row = actions_df_country['Country'].value_counts().to_dict()
ratings_per_country = pd.DataFrame(count_row, index = ['#ratings'])
print("# Ratings: " + str(ratings_per_country.iloc[0].sum()))
print(ratings_per_country)

#calculate fraction of ratings per country
ratings_per_country = ratings_per_country / (ratings_per_country.iloc[0].sum() / 100)
print(ratings_per_country)




#---------- START GENRE COUNTRY CALCULATIONS ----------
# extract unique genres for analyse
unique_genres = genre_movie_list['genres'].unique()
unique_genres = np.append(unique_genres, ['lang', 'k'])
# create result list
predicted_counts = pd.DataFrame(columns=unique_genres)

# extract recommended movies per model and add it to genre list
for i, row in predicted_movies.iterrows():
    recommendations = row['recommended_movies'].to_dataframe()
    #join genre movie list with recommended movies
    recommendations_with_genre = pd.merge(recommendations, genre_movie_list, left_on='item_id', right_on='movie_id', how="left")
    # count the movies per genre and add to result list
    row_result = recommendations_with_genre['genres'].value_counts().to_dict()
    row_result['k'] = row['k']
    row_result['lang'] = row['lang']
    predicted_counts = predicted_counts.append(row_result, ignore_index=True)
            
# fill null values     
predicted_counts = predicted_counts.fillna(0)
print(predicted_counts)


# sums associated movies per model
sum_rows = predicted_counts.sum(axis='columns')

print(sum_rows)
#extract unique genres
unique_genres = genre_movie_list['genres'].unique()

for genre in unique_genres:
    for i, row in predicted_counts.iterrows():
        #calcualte fraction
        predicted_counts.at[i, genre] = predicted_counts.iloc[i][genre] / (sum_rows[i] /100)

print(predicted_counts)


# used for genre statistics
actions_df_genre = actions_df.copy()

# calculate number of ratings per genre, first join ratings list with genre - movie list
actions_df_genre = pd.merge(actions_df_genre, genre_movie_list, left_on='item_id', right_on='movie_id', how="left")
count_row = actions_df_genre['genres'].value_counts().to_dict()
ratings_per_genre = pd.DataFrame(count_row, index = ['#ratings'])
print("# Ratings: " + str(ratings_per_genre.iloc[0].sum()))
print(ratings_per_genre)



#calculate fraction of ratings per genre
ratings_per_genre = ratings_per_genre / (ratings_per_genre.iloc[0].sum() / 100)
print(ratings_per_genre)



