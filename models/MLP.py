from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.optimizers import  Adam, SGD
from typing import Tuple


class MLP:
    
    def __init__(self, config: dict, num_users: int, num_items: int) -> None:
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.output_layer_activation = config['output_layer_activation']
        self.layers = config['layers']
        self.num_layers = config['num_layers']
        self.embedding_out_dim = config['embedding_out_dim']
        self.model_parameters_path = config['model_parameters_path']
        self.user_input = Input(shape=(1,), name='user_input')
        self.item_input = Input(shape=(1,), name='item_input')

        self.num_users = num_users
        self.num_items = num_items

        self.opt = config["optimizer"]
        self.loss = config["loss"]
        self.metrics = config["metrics"]

    def create_embeddings(self) -> Tuple[Embedding, Embedding]:
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_out_dim, embeddings_initializer='normal', name='mlp_user_embedding')
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_out_dim, embeddings_initializer='normal', name='mlp_item_embedding')
        return user_embedding, item_embedding
    
    def create_vector(self, user_embedding: Embedding, item_embedding: Embedding) -> Concatenate:
        mlp_user_latent = Flatten()(user_embedding(self.user_input))
        mlp_item_latent = Flatten()(item_embedding(self.item_input))
        return Concatenate()([mlp_user_latent, mlp_item_latent])
    
    def build_model(self) -> Model:
        user_embedding, item_embedding = self.create_embeddings()

        mlp_vector = self.create_vector(user_embedding, item_embedding)

        for i, n_units in enumerate(self.layers):
            mlp_vector = Dense(n_units, activation='relu', name=f'mlp_layer_{i}')(mlp_vector)

        out_layer = Dense(1, activation=self.output_layer_activation, name='out_layer')(mlp_vector)
        model = Model(inputs=[self.user_input, self.item_input], outputs=out_layer, name='multi_layer_perceptron')
        #self.save_model_summary(model)

        return model
    
    def compile_model(self, model: Model) -> Model:
        if self.opt == "adam":
            opt = Adam(learning_rate=self.learning_rate)
        elif self.opt == "sgd":
            opt = SGD(learning_rate=self.learning_rate)

        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        return model
    

    def save_model_summary(self, model: Model) -> None:
        with open('../data/summaries/mlp_model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
