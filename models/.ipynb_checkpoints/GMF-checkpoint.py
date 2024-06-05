import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Multiply
from keras.optimizers import Adam, SGD
from sklearn.metrics import mean_absolute_error
from typing import Tuple

class GMF:
    def __init__(self, config: dict, num_users: int, num_items: int) -> None:
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.output_layer_activation = config['output_layer_activation']
        self.factors = config['factors']
        self.model_parameters_path = config['model_parameters_path']

        self.user_input = Input(shape=(1,), name='user_input')
        self.item_input = Input(shape=(1,), name='item_input')

        self.num_users = num_users
        self.num_items = num_items

        self.opt = config["optimizer"]
        self.loss = config["loss"]
        self.metrics = config["metrics"]

    def create_embeddings(self) -> Tuple[Embedding, Embedding]:
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.factors, name='gmf_user_embedding')
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.factors, name='gmf_item_embedding')
        return user_embedding, item_embedding
    
    def create_vector(self, user_embedding: Embedding, item_embedding: Embedding) -> Multiply:
        gmf_user_latent = Flatten()(user_embedding(self.user_input))
        gmf_item_latent = Flatten()(item_embedding(self.item_input))
        return Multiply()([gmf_user_latent, gmf_item_latent])
    
    def build_model(self) -> Model:
        user_embedding, item_embedding = self.create_embeddings()

        gmf_vector = self.create_vector(user_embedding, item_embedding)

        out_layer = Dense(1, activation=self.output_layer_activation, name='out_layer')(gmf_vector)
        model = Model(inputs=[self.user_input, self.item_input], outputs=out_layer, name='generalized_matrix_factorization')
        self.save_model_summary(model)

        return model
       
    def compile_model(self, model: Model) -> Model:
        if self.opt == "adam":
            opt = Adam(learning_rate=self.learning_rate)
        elif self.opt == "sgd":
            opt = SGD(learning_rate=self.learning_rate)
 
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        return model
    
    def save_model_summary(self, model: Model) -> None:
        with open('./models/summaries/gmf_model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))