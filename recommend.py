from models.NeuMF import NeuMF
from utils.read_files import read_yaml_file, read_jokes_csv_file, read_ratings_csv_file
import numpy as np

RATINGS_PATH = './data/preprocessed_data/ratings/ratings_preprocessed_ml.csv'
JOKES_PATH = './data/preprocessed_data/jokes/jokes_preprocessed.csv'

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

    user_id = np.full(num_items, user_id)
    jokes_ids = np.arange(1, num_items+1)
    pred = neumf_model.predict([user_id, jokes_ids])
    top_k = np.argsort(pred, axis=0).reshape(-1)[::-1][:num_recommendations]
    top_k_jokes = jokes.iloc[top_k]
    return top_k_jokes.values

def recommend_items_for_new_user(num_recommendations=10):
    ratings = read_ratings_csv_file()
    jokes = read_jokes_csv_file()
    
    most_popular_jokes_idx = ratings.value_counts('joke_id').sort_values(ascending=False)
    top_k_popular_jokes = most_popular_jokes_idx.index[:num_recommendations]
    pop_jokes = jokes.iloc[top_k_popular_jokes]
    return pop_jokes.values







def main():
    data = read_yaml_file()
    ratings = read_ratings_csv_file()
    jokes = read_jokes_csv_file()

    num_users = ratings['user'].nunique()
    num_items = ratings['joke_id'].nunique()

    neumf = NeuMF(data['NeuMF'], num_users, num_items)
    neumf_model = neumf.build_model()
    neumf_model = neumf.compile_model(neumf_model)

    neumf_model.load_weights('./models/weights/neumf_weights.h5')





if __name__ == '__main__':
    main()