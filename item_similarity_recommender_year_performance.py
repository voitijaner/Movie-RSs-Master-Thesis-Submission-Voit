from __future__ import division
import graphlab as gl
import pandas as pd
from sklearn import cross_validation


# load ratings file
actions_df = pd.read_csv('data/movielense/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], encoding="utf-8")
# load and transform final movie list with years
year_movie_list = pd.read_csv('data/movielense/final_movie_genre_year_county_list.csv', usecols=['movieId', 'year'])
year_movie_list = year_movie_list.drop_duplicates()
year_movie_list = year_movie_list.rename(columns={'movieId' : 'movie_id'})


# ----------- PREPROCESSING -----------


# Remove movies which are not in the final movie list
print("Initial ratings: " + str(len(actions_df)))
movie_list = year_movie_list['movie_id'].tolist()
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

# create the year bins and store in final_year_list
year_bins = [[1950, 1980], [1980, 1990], [1990, 2010]]
final_year_list = []
for year in year_bins:
    year_movies = year_movie_list.loc[(year_movie_list['year'] >= year[0]) & (year_movie_list['year'] < year[1])]['movie_id'].values
    final_year_list.append([year, year_movies])



# year statistics
for year in final_year_list:
    temp = actions_df[actions_df['item_id'].isin(year[1])]
    print(year[0])
    print("ratings: " + str(len(temp)))
    print("items: " + str(len(temp['item_id'].unique())))
    print("users: " + str(len(temp['user_id'].unique())))

#add placeholder for all data
final_year_list.append(['all', ['']])


# ----------- CREATE RS MODELS -----------


def create_RS(lang, k, training_data, nearest_items_sf_year):
    """Create the item_similarity_recommender models and returns it."""
    model = gl.item_similarity_recommender.create(training_data, user_id='user_id', item_id='item_id', similarity_type="cosine", target="rating", only_top_k = k, nearest_items=nearest_items_sf_year)
    return model


model_list = []

# for each year, the recommender models for the best k's will be created
for year in final_year_list:
    # if models are created over whole dataset: all ratings are used
    if year[0] == 'all':
        actions_df_year = actions_df
    else:
        # extract year-bins ratings and remove users with less than 50 ratings
        actions_df_year = actions_df[actions_df['item_id'].isin(year[1])]
        actions_df_year = actions_df_year.groupby('user_id').filter(lambda x: len(x) >= 50)

    print("------------------------------------START New Year " +" ".join(str(x) for x in year[0])+ " ------------------------------------")
    for i in range(3):
        #split in train and test data sets, each langauge version is is trained with k between 1 and 10 on the same splits
        train, test = cross_validation.train_test_split(actions_df_year, test_size=0.15, stratify=actions_df_year['user_id']) 
        training_data = gl.SFrame(train)
        validation_data = gl.SFrame(test)
        for l in ["de", "fr", "it", "ru", "en"]:
            #load 50 most similar items for the language version
            nearest_items_df = pd.read_csv('data/similar_movies/50_nearest_items_lang='+l+'.csv', usecols=['movie_id','similar','score'])
            if year[0] == 'all':
                nearest_items_df_year = nearest_items_df
            else:
                # load similar movies for each year bin, enouch movies contained per year bin, so its not neccesary
                # to load similar movie lists specificly created for each year bin (in conctrast to genre)
                nearest_items_df_year = nearest_items_df[nearest_items_df['movie_id'].isin(year[1])]
                nearest_items_df_year = nearest_items_df_year[nearest_items_df_year['similar'].isin(year[1])]
            
            # similar movie_id and the similar movies are seen as float, convert to int
            nearest_items_df_year['similar'] = nearest_items_df_year['similar'].astype(int)
            nearest_items_df_year['movie_id'] = nearest_items_df_year['movie_id'].astype(int)
            nearest_items_df_year = nearest_items_df_year.rename(columns={"movie_id": "item_id"})
            nearest_items = gl.SFrame(nearest_items_df_year)
            
            # set k to the best per language
            if l == "de":
                k = 3
            if l == "fr":
                k = 3
            if l == "it":
                k = 2
            if l == "ru":
                k = 6
            if l == "en":
                k = 2

            model = create_RS(l, k , training_data, nearest_items)
            model_list.append([model, validation_data, l, year[0], k])



# convert model_list into DataFrame for later use
model_list_df = pd.DataFrame(model_list, columns = ["model", "validation_data", "lang", "year", "k"])
model_list_df


# ----------- EVALUATE RS MODELS -----------


# resulting df for precision / recall extraction
precision_recall_df = pd.DataFrame(columns=['precision', 'recall', 'year', 'k', 'lang'])
for i, row in model_list_df.iterrows():
    validation_data = row['validation_data']
    lang = row['lang']
    k = row['k']
    year = row['year']
    model = row['model']
    # extract relevant movies and evaluate the RS with them
    validation_data_by_year_and_relevant_items = validation_data.filter_by([4,5], 'rating')
    predictions_by_year = model.evaluate(validation_data_by_year_and_relevant_items, verbose=False)
    precision_recall_by_year = predictions_by_year['precision_recall_overall'].filter_by([10], 'cutoff')
    # add to precision - recall results list
    result_row = {'precision':precision_recall_by_year['precision'][0], 'recall':precision_recall_by_year['recall'][0], 'year':year[0], 'k':k, 'lang':lang}
    precision_recall_df = precision_recall_df.append(result_row, ignore_index=True)


# list wich containes the averaged precision / recall and F1 scores for each model - langauge version
model_results_df = pd.DataFrame(columns=['mean_precision', 'mean_recall', 'F1', 'k', 'lang', 'year'])
for lang in precision_recall_df["lang"].unique():
    for year in precision_recall_df["year"].unique():
        # extract the results for the each language and year pair
        prec_rows = precision_recall_df.loc[(precision_recall_df["lang"] == lang)  & (precision_recall_df["year"] == year)]
        presicion_per_model = 0
        recall_per_model = 0
        k = 0
        #calcualte avg. precision, recall and f1 score
        for i, row in prec_rows.iterrows():
            k = row['k']
            presicion_per_model = presicion_per_model + row['precision']
            recall_per_model = recall_per_model + row['recall']
        avg_presicion_per_model = presicion_per_model / len(prec_rows)
        avg_recall_per_model = recall_per_model / len(prec_rows)
        f1 = 2*avg_presicion_per_model*avg_recall_per_model/(avg_presicion_per_model + avg_recall_per_model)
        # add to model results list
        result_row = {'mean_precision':avg_presicion_per_model, 'mean_recall':avg_recall_per_model, 'F1':f1, 'k':k , 'lang':lang, 'year':year}
        model_results_df = model_results_df.append(result_row, ignore_index = True)


# ----------- OUTPUT RS MODEL RESULTS -----------


print(model_results_df.sort_values(by=['year', 'F1'], ascending=False))

#calculate the difference of the overall performance to the model results for each year
results_with_diff = model_results_df.copy()
for i, row in results_with_diff.iterrows():
    precision_all = results_with_diff.loc[(results_with_diff['year']=='a') & (results_with_diff['lang']==row['lang'])]['mean_precision'].values[0]
    recall_all = results_with_diff.loc[(results_with_diff['year']=='a') & (results_with_diff['lang']==row['lang'])]['mean_recall'].values[0]
    F1_all = results_with_diff.loc[(results_with_diff['year']=='a') & (results_with_diff['lang']==row['lang'])]['F1'].values[0]
    precision_diff = row['mean_precision'] -  precision_all
    recall_diff = row['mean_recall'] -  recall_all
    F1_diff = row['F1'] - F1_all
    results_with_diff.at[i, 'precision_diff'] = 100 / precision_all * precision_diff
    results_with_diff.at[i, 'recall_diff'] = 100 / recall_all * recall_diff
    results_with_diff.at[i, 'F1_diff'] = 100 / F1_all * F1_diff

print(results_with_diff.sort_values(by=['year', 'F1'], ascending=False))

