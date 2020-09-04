from __future__ import division
import graphlab as gl
import pandas as pd
from sklearn import cross_validation


# load ratings file
actions_df = pd.read_csv('data/movielense/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], encoding="utf-8")
# load and transform final movie list with genres
genre_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movieId', 'genres'])
genre_movie_list = genre_movie_list.drop_duplicates()
genre_movie_list = genre_movie_list.rename(columns={'movieId' : 'movie_id'})


# ----------- PREPROCESSING -----------

# Remove movies which are not in the final movie list
print("Initial ratings: " + str(len(actions_df)))
movie_list = genre_movie_list['movie_id'].tolist()
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



# remove movies from genre_list which are no longer in ratings data set (after preprocessing)
boolean_series = genre_movie_list.movie_id.isin(actions_df['item_id'].unique())
genre_movie_list = genre_movie_list[boolean_series]

# create final genre list with a movie list per genre
final_genre_list = []
for genre in genre_movie_list['genres'].unique():
    genre_movies = genre_movie_list[genre_movie_list['genres'] == genre]['movie_id'].values
    if len(genre_movies) > 100:
        final_genre_list.append([genre, genre_movies])

# genre statistics
for row in final_genre_list:
    print(row[0])
    print("items: " + str(len(row[1])))
    temp = actions_df[actions_df['item_id'].isin(row[1])]
    temp = temp.groupby('user_id').filter(lambda x: len(x) >= 50)
    print("ratings: " + str(len(temp)))
    print("users: " + str(len(temp['user_id'].unique())))

#add placeholder for all data
final_genre_list.append(['all', ['']])

# ----------- CREATE RS MODELS -----------

def create_RS(lang, k, training_data, nearest_items_sf_genre):
    """Create the item_similarity_recommender models and returns it."""
    model = gl.item_similarity_recommender.create(training_data, similarity_type="cosine", user_id='user_id', item_id='item_id', target="rating", only_top_k = k, nearest_items=nearest_items_sf_genre)
    return model



model_list = []

# for each genre, the recommender models for the best k's will be created
for genre in final_genre_list:
    # if models are created over whole dataset: all ratings are used
    if genre[0] == 'all':
        actions_df_genre = actions_df
    else:
        # extract genre ratings and remove users with less than 50 ratings
        actions_df_genre = actions_df[actions_df['item_id'].isin(genre[1])]
        actions_df_genre = actions_df_genre.groupby('user_id').filter(lambda x: len(x) >= 50)

    print("------------------------------------START GENRE " + genre[0] + "------------------------------------")
    for i in range(3):
        #split in train and test data sets, each langauge version is is trained with k between 1 and 10 on the same splits
        train, test = cross_validation.train_test_split(actions_df_genre, test_size=0.15, stratify=actions_df_genre['user_id']) 
        training_data = gl.SFrame(train)
        validation_data = gl.SFrame(test)
        for lang in ["de", "en", "it", "ru", "fr"]:
            # load similar items for all data or genre specific lists
            if genre[0] == 'all':
                nearest_items_df_genre = pd.read_csv('data/similar_movies/50_nearest_items_lang='+lang+'.csv', usecols=['movie_id','similar','score'])
            else:
                nearest_items_df_genre = pd.read_csv('data/similar_movies/genre/10_nearest_items_'+genre[0]+'_lang='+lang+'.csv', usecols=['movie_id','similar','score'])
            
            # similar movie_id and the similar movies are seen as float, convert to int
            nearest_items_df_genre['similar'] = nearest_items_df_genre['similar'].astype(int)
            nearest_items_df_genre['movie_id'] = nearest_items_df_genre['movie_id'].astype(int)            
            nearest_items_df_genre = nearest_items_df_genre.rename(columns={"movie_id": "item_id"})

                
            nearest_items = gl.SFrame(nearest_items_df_genre)
            
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
                
            model = create_RS(l, k , training_data, nearest_items)
            model_list.append([model, validation_data, lang, genre[0], k])


# convert model_list into DataFrame for later use
model_list_df = pd.DataFrame(model_list, columns = ["model", "validation_data", "lang", "genre", "k"])
model_list_df


# ----------- EVALUATE RS MODELS -----------

# resulting df for precision / recall extraction
precision_recall_df = pd.DataFrame(columns=['precision', 'recall', 'genre', 'k', 'lang'])
count = 0
for i, row in model_list_df.iterrows():
    count += 1
    print(count)
    validation_data = row['validation_data']
    lang = row['lang']
    k = row['k']
    genre = row['genre']
    model = row['model']
    # extract relevant movies and evaluate the RS with them
    validation_data_by_genre_and_relevant_items = validation_data.filter_by([4,5], 'rating')
    predictions_by_genre = model.evaluate(validation_data_by_genre_and_relevant_items, verbose=False)
    precision_recall_by_genre = predictions_by_genre['precision_recall_overall'].filter_by([10], 'cutoff')
    # add to precision - recall results list
    result_row = {'precision':precision_recall_by_genre['precision'][0], 'recall':precision_recall_by_genre['recall'][0], 'genre':genre, 'k':k, 'lang':lang}
    precision_recall_df = precision_recall_df.append(result_row, ignore_index=True)



# list wich containes the averaged precision / recall and F1 scores for each model - langauge version
model_results_df = pd.DataFrame(columns=['mean_precision', 'mean_recall', 'F1', 'k', 'lang', 'genre'])
for lang in precision_recall_df["lang"].unique():
    for genre in precision_recall_df["genre"].unique():
        # extract the models for the each language and genre pair
        prec_rows = precision_recall_df.loc[(precision_recall_df["lang"] == lang) & (precision_recall_df["genre"] == genre)]
        presicion_per_model = 0
        recall_per_model = 0
        k = 0
        #calcualte avg. precision, recall and f1 score
        for i, row in prec_rows.iterrows():
            k = row['k']
            presicion_per_model = presicion_per_model + row['precision']
            recall_per_model = recall_per_model + row['recall']
            if genre == "all":
                print(lang)
                print("Precision: " + str(row['precision']) + " , Recall: " + str(row['recall']) + " , F1: " + str(2*row['precision']*row['recall']/(row['precision']+row['recall'])))
        avg_presicion_per_model = presicion_per_model / len(prec_rows)
        avg_recall_per_model = recall_per_model / len(prec_rows)
        f1 = 2*avg_presicion_per_model*avg_recall_per_model/(avg_presicion_per_model + avg_recall_per_model)
        # add to model results list
        result_row = {'mean_precision':avg_presicion_per_model, 'mean_recall':avg_recall_per_model, 'F1':f1, 'k':k, 'lang':lang, 'genre':genre}
        model_results_df = model_results_df.append(result_row, ignore_index = True)


# ----------- OUTPUT RS MODEL RESULTS -----------

print(model_results_df.sort_values(by=['genre', 'F1'], ascending=False))


#calculate the difference of the overall performance to the model results for each year
results_with_diff = model_results_df.copy()
#results_with_diff =  model_results_df[model_results_df['k']==4]
for i, row in results_with_diff.iterrows():
    precision_all = results_with_diff.loc[(results_with_diff['genre']=='all') & (results_with_diff['lang']==row['lang'])]['mean_precision'].values[0]
    recall_all = results_with_diff.loc[(results_with_diff['genre']=='all') & (results_with_diff['lang']==row['lang'])]['mean_recall'].values[0]
    F1_all = results_with_diff.loc[(results_with_diff['genre']=='all') & (results_with_diff['lang']==row['lang'])]['F1'].values[0]
    precision_diff = row['mean_precision'] -  precision_all
    recall_diff = row['mean_recall'] -  recall_all
    F1_diff = row['F1'] - F1_all
    results_with_diff.at[i, 'precision_diff'] = 100 / precision_all * precision_diff
    results_with_diff.at[i, 'recall_diff'] = 100 / recall_all * recall_diff
    results_with_diff.at[i, 'F1_diff'] = 100 / F1_all * F1_diff


print(results_with_diff.sort_values(by=['genre', 'F1'], ascending=False))

