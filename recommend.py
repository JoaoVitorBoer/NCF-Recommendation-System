from models.NeuMF import NeuMF
from utils.read_files import read_yaml_file, read_jokes_csv_file, read_ratings_csv_file
import numpy as np

def recommend_items_for_existing_user(user_id, num_recommendations=10):
    data = read_yaml_file()
    ratings = read_ratings_csv_file()
    jokes = read_jokes_csv_file()

    num_users = ratings['user'].nunique()
    num_items = ratings['joke_id'].nunique()

    neumf = NeuMF(data['NeuMF'], num_users, num_items)
    neumf_model = neumf.build_model()
    neumf_model = neumf.compile_model(neumf_model)

    neumf_model.load_weights('./data/weights/neumf.h5')

    user_id_list = np.full(num_items, user_id) # user_id is repeated for all jokes
    jokes_ids = np.arange(1, num_items+1) # joke_id is from 1 to num_items
    pred = neumf_model.predict([user_id_list, jokes_ids]) 
    top_k = np.argsort(pred, axis=0).reshape(-1)[::-1][:num_recommendations]
    top_k_jokes = jokes.iloc[top_k] # get the top k jokes text
    return top_k_jokes.values

def recommend_items_for_new_user(num_recommendations=10):
    ratings = read_ratings_csv_file()
    jokes = read_jokes_csv_file()
    
    most_popular_jokes_idx = ratings.value_counts('joke_id').sort_values(ascending=False) # get the most popular jokes
    top_k_popular_jokes = most_popular_jokes_idx.index[:num_recommendations]
    pop_jokes = jokes.iloc[top_k_popular_jokes]
    return pop_jokes.values


def recommend_new_item_for_existing_user(user_id):
    data = read_yaml_file()
    ratings = read_ratings_csv_file()
    jokes = read_jokes_csv_file()

    num_users = ratings['user'].nunique()
    num_items = ratings['joke_id'].nunique()

    neumf = NeuMF(data['NeuMF'], num_users, num_items)
    neumf_model = neumf.build_model()
    neumf_model = neumf.compile_model(neumf_model)

    neumf_model.load_weights('./data/weights/neumf.h5')
    
    #Finding the jokes that the user has not rated 
    unique_jokes_id = ratings['joke_id'].unique()
    user_rated_jokes = ratings[ratings['user'] == user_id]['joke_id'].values
    user_unrated_jokes = np.setdiff1d(unique_jokes_id, user_rated_jokes)
    if user_unrated_jokes.size == 0:
        return recommend_items_for_existing_user(user_id, num_recommendations=1)
    
    predictions = []
    for joke_id in user_unrated_jokes:
        pred = neumf_model.predict([np.array([user_id]), np.array([joke_id])])
        predictions.append(pred)
   
    top_joke_idx = np.argmax(predictions)
    top_joke = jokes.iloc[top_joke_idx]
    return top_joke.values




