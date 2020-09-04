from __future__ import division
import graphlab as gl
import pandas as pd
from sklearn import cross_validation
import numpy as np


# load ratings file
actions_df = pd.read_csv('data/movielense/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], encoding="utf-8")
# load and transform final movie list
final_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movieId'])
final_movie_list = final_movie_list.drop_duplicates()
final_movie_list = final_movie_list.rename(columns={'movieId' : 'movie_id'})


# ----------- PREPROCESSING -----------


# Remove movies which are not in the final movie list
print("Initial ratings: " + str(len(actions_df)))
movie_list = final_movie_list['movie_id'].tolist()
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


# ----------- CREATE RS MODELS -----------


def create_RS(lang, k, training_data, nearest_items_sf):
    """Create the item_similarity_recommender models and returns it."""
    model = gl.item_similarity_recommender.create(training_data, similarity_type="cosine", user_id='user_id', item_id='item_id', target="rating", only_top_k = k, nearest_items=nearest_items_sf)
    return model

model_list = []

#three runs for cross validation
for i in range(3):
    #split in train and test data sets, each langauge version is is trained with k between 1 and 10 on the same splits
    train, test = cross_validation.train_test_split(actions_df, test_size=0.15, stratify=actions_df['user_id']) 
    training_data = gl.SFrame(train)
    validation_data = gl.SFrame(test)
    for l in ["de", "en", "it", "ru", "fr"]:
        #load 50 most similar items for the language version
        nearest_items_df = pd.read_csv("data/similar_movies/50_nearest_items_lang="+l+".csv", usecols=['movie_id','similar','score'])
        # similar movie_id and the similar movies are seen as float, convert to int
        nearest_items_df['similar'] = nearest_items_df['similar'].astype(int)
        nearest_items_df['movie_id'] = nearest_items_df['movie_id'].astype(int)
        nearest_items_df = nearest_items_df.rename(columns={"movie_id": "item_id"})
        
        nearest_items = gl.SFrame(nearest_items_df)
        
        # k = consider the k most similar movies for the rs
        for k in range(1,11):
            model = create_RS(l, k , training_data, nearest_items)
            model_list.append([model, validation_data, l, k])

# convert model_list into DataFrame for later use
model_list_df = pd.DataFrame(model_list, columns = ["model", "validation_data", "lang", "k"])
model_list_df


# ----------- EVALUATE RS MODELS -----------


# resulting df for precision / recall extraction
precision_recall_df = pd.DataFrame(columns=['precision', 'recall', 'k', 'lang'])
for i, row in model_list_df.iterrows():
    validation_data = row['validation_data']
    lang = row['lang']
    k = row['k']
    model = row['model']
    # extract relevant movies and evaluate the RS with them
    validation_data = validation_data.filter_by([4,5], 'rating')
    predictions = model.evaluate(validation_data, verbose=False)
    precision_recall = predictions['precision_recall_overall'].filter_by([10], 'cutoff')
    # add to precision - recall results list
    result_row = {'precision':precision_recall['precision'][0], 'recall':precision_recall['recall'][0], 'k':k, 'lang':lang}
    precision_recall_df = precision_recall_df.append(result_row, ignore_index=True)


# list wich containes the averaged precision / recall and F1 scores for each model - langauge version
model_results_df = pd.DataFrame(columns=['mean_precision', 'mean_recall', 'F1', 'k', 'lang'])
for lang in precision_recall_df["lang"].unique():
    for k in precision_recall_df["k"].unique():
        # extract the results for the each language and k pair
        prec_rows = precision_recall_df.loc[(precision_recall_df["lang"] == lang) & (precision_recall_df["k"] == k)]
        presicion_per_model = 0
        recall_per_model = 0
        #calcualte avg. precision, recall and f1 score
        for i, row in prec_rows.iterrows():
            presicion_per_model = presicion_per_model + row['precision']
            recall_per_model = recall_per_model + row['recall']
        avg_presicion_per_model = presicion_per_model / len(prec_rows)
        avg_recall_per_model = recall_per_model / len(prec_rows)
        f1 = 2*avg_presicion_per_model*avg_recall_per_model/(avg_presicion_per_model + avg_recall_per_model)
        # add to model_results list
        result_row = {'mean_precision':avg_presicion_per_model, 'mean_recall':avg_recall_per_model, 'F1':f1, 'k':k, 'lang':lang}
        model_results_df = model_results_df.append(result_row, ignore_index = True)


# ----------- OUTPUT RS MODEL RESULTS -----------


print(model_results_df.sort_values(by=['lang', 'F1'], ascending=False))


print(model_results_df.sort_values(by=['F1'], ascending=False))




