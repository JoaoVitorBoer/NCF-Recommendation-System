import streamlit as st
import numpy as np
import pandas as pd
from recommend import recommend_items_for_existing_user, recommend_items_for_new_user
 
st.title('Jokes Recommendation')

st.header('Recommend a list of jokes for an old user')
user_id_existing = st.text_input('User ID:')
num_recommendations_existing = st.number_input('Nº of recomendations:', min_value=1, max_value=10, value=5)
if st.button('Recommend for old user'):
    clear_output = st.checkbox('Clear Output')
    if not clear_output:
        if int(user_id_existing) > 0 and int(user_id_existing) <= 73421:
            recommended_items_existing = recommend_items_for_existing_user(int(user_id_existing), num_recommendations_existing)
            st.title('Jokes recommended:')
            for i, joke in enumerate(recommended_items_existing):
                st.title(f'Joke {i+1}')
                st.write(str(*joke))
        else:
            st.write('Please, insert user ID in range 0 to 73421.')

# Scenario 2: Recommend items for a new user
st.header('Recommend a list of itens for a new user')
num_recommendations_new = st.number_input('Nº of recomendations for new user:', min_value=1, max_value=10, value=5)
if st.button('Recommend to new user'):
    clear_output_cold_start = st.checkbox('Clear Output')
    if not clear_output_cold_start:
        recommended_items_new = recommend_items_for_new_user(num_recommendations_new)
        st.title('Cold Start Problem')
        st.write('New user has no ratings. We recommend the most popular items.') 
        st.title('Jokes recommended:')
        for i, joke in enumerate(recommended_items_new):
                st.title(f'Joke {i+1}')
                st.write(str(*joke))

# # Scenario 3: Recommend a new item for an existing user
# st.header('Recommend a new item for an existing user')
# user_id_existing_item = st.text_input('User ID for a new item:')
# if st.button('Run'):
#     if int(user_id_existing_item) > 0 and int(user_id_existing_item) <= 73421:
#         recommended_new_item = recommend_new_item_for_existing_user(user_id_existing_item)
#         st.write('Item recommended:')
#         st.write(recommended_new_item)
#     else:
#         st.write('Please, insert user ID in range 1 to 73421')

