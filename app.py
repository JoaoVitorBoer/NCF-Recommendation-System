import streamlit as st
import numpy as np
import pandas as pd


# Function to get recommendations for an existing user
def recommend_items_for_existing_user(user_id, num_recommendations=10):
    pass

# Function to get recommendations for a new user
def recommend_items_for_new_user(num_recommendations=10):
    pass
# Function to recommend a new item for an existing user
def recommend_new_item_for_existing_user(user_id):
    pass
 


st.title('Jokes Recommendation')

st.header('Recommend a list of itens for an old user')
user_id_existing = st.text_input('User ID:')
num_recommendations_existing = st.number_input('Nº of recomendations:', min_value=1, max_value=10, value=5)
if st.button('Recommend for old user'):
    if int(user_id_existing) > 0 and int(user_id_existing) <= 73421:
        recommended_items_existing = recommend_items_for_existing_user(user_id_existing, num_recommendations_existing)
        st.write('Items recommended:')
        for x in ['a', 'b', 'b', 'b', 'b', 'b', 'b']:
            st.write(x)
    else:
        st.write('Please, insert user ID in range 1 to 73421.')

# Scenario 2: Recommend items for a new user
st.header('Recommend a list of itens for a new user')
num_recommendations_new = st.number_input('Nº of recomendations for new user:', min_value=1, max_value=10, value=5)
if st.button('Recommend to new user'):
    recommended_items_new = recommend_items_for_new_user(num_recommendations_new)
    st.caption('Cold Start Problem')
    st.write('Itens recommended:')
    st.write(recommended_items_new)

# Scenario 3: Recommend a new item for an existing user
st.header('Recommend a new item for an existing user')
user_id_existing_item = st.text_input('User ID for a new item:')
if st.button('Run'):
    if int(user_id_existing_item) > 0 and int(user_id_existing_item) <= 73421:
        recommended_new_item = recommend_new_item_for_existing_user(user_id_existing_item)
        st.write('Item recommended:')
        st.write(recommended_new_item)
    else:
        st.write('Please, insert user ID in range 1 to 73421')

