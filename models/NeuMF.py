from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Multiply
from keras.optimizers import  Adam, SGD
from typing import Tuple
import numpy as np
from models.MLP import MLP
from models.GMF import GMF


class NeuMF:
    def __init__(self, config: dict, num_users: int, num_items: int) -> None:
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.output_layer_activation = config['output_layer_activation']

        self.mlp_layers = config['MLP']['layers']
        self.mlp_num__layers = config['MLP']['num_layers']
        self.mlp_embedding_out_dim = config['MLP']['embedding_out_dim']

        self.gmf_factors = config['GMF']['factors']
        
        self.user_input = Input(shape=(1,), name='user_input')
        self.item_input = Input(shape=(1,), name='item_input')
        
        self.opt = config["optimizer"]
        self.loss = config["loss"]
        self.metrics = config["metrics"]


    def get_embeddings(self, model_type: str) -> Tuple[Embedding, Embedding]:
        out_dim = self.mlp_embedding_out_dim if model_type == 'mlp' else self.gmf_factors
        user_embedding = Embedding(input_dim=self.num_users, output_dim=out_dim, name=f'{model_type}_user_embedding')
        item_embedding = Embedding(input_dim=self.num_items, output_dim=out_dim, name=f'{model_type}_item_embedding')
        return user_embedding, item_embedding

    def create_vector(self, user_embedding: Embedding, item_embedding: Embedding, model_type: str) -> Concatenate | Multiply:
        user_latent = Flatten()(user_embedding(self.user_input))
        item_latent = Flatten()(item_embedding(self.item_input))
        return Concatenate()([user_latent, item_latent]) if model_type == 'mlp' else Multiply()([user_latent, item_latent])
    
    def build_model(self) -> Model:
        mlp_user_embedding, mlp_item_embedding = self.get_embeddings('mlp')
        gmf_user_embedding, gmf_item_embedding = self.get_embeddings('gmf')

        mlp_vector = self.create_vector(mlp_user_embedding, mlp_item_embedding, 'mlp')
        gmf_vector = self.create_vector(gmf_user_embedding, gmf_item_embedding, 'gmf')

        for i, n_units in enumerate(self.mlp_layers):
            mlp_vector = Dense(n_units, activation='relu', name=f'mlp_layer_{i}')(mlp_vector)
        
        pred_vector = Concatenate()([mlp_vector, gmf_vector])
        out_layer = Dense(1, activation=self.output_layer_activation, name='out_layer')(pred_vector)

        model = Model(inputs=[self.user_input, self.item_input], outputs=out_layer, name='neural_matrix_factorization')
        self.save_model_summary(model)

        return model
    
    def compile_model(self, model: Model) -> Model:
        """As we are always using a pretrained model, we wont use Adam because it
          needs to save momentum information for updating parameters properly. 
          But the code is here just for testing puporses"""
        
        if self.opt == "adam":
            opt = Adam(learning_rate=self.learning_rate)
        elif self.opt == "sgd":
            opt = SGD(learning_rate=self.learning_rate)
 
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        return model
    
    def load_pretrain_weights(self, neumf_model: Model, mlp_model: Model, gmf_model: Model) -> Model:
        """Load pretrained weights from GMF and MLP models to the NeuMF model
           alpha is a hyperparameter that can be tuned to give more importance to one model over the other.
           In this case we are using 0.5 to give the same importance to both models. And also 0.5 is used in the paper
        """
        layers_to_be_loaded = neumf_model.layers[2:-1] #Ignore user_input, item_input and prediction layers layers
        for layer in layers_to_be_loaded: 
            if layer.name in [layer.name for layer in mlp_model.layers]:
                layer.set_weights(mlp_model.get_layer(layer.name).get_weights())
            elif layer.name in [layer.name for layer in gmf_model.layers]:
                layer.set_weights(gmf_model.get_layer(layer.name).get_weights())
        
        gmf_pred = gmf_model.get_layer('out_layer').get_weights()
        mlp_pred = mlp_model.get_layer('out_layer').get_weights()
        
        alpha = 0.5
        new_weights = np.concatenate((alpha*gmf_pred[0], (1-alpha)*mlp_pred[0]), axis=0)
        new_bias = alpha*gmf_pred[1] + (1-alpha)*mlp_pred[1]
        neumf_model.get_layer('out_layer').set_weights([new_weights, new_bias   ])

        return neumf_model        
        
    def save_model_summary(self, model: Model) -> None:
        with open('../data/summaries/neumf_model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def train_MLP(self, mlp_model: Model, user_train,  item_train, y_train, eval_data):
        print('-------- Training MLP --------')
        user_eval, item_eval, rating_eval = eval_data
        history = mlp_model.fit([user_train, item_train],
                      y_train, 
                      validation_data=([user_eval, item_eval], rating_eval), 
                      epochs=self.config['MLP']['epochs'], 
                      batch_size=self.config['MLP']['batch_size'], 
                      verbose=1)
        mlp_model.save(self.config['MLP']['model_parameters_path'])

    def train_GMF(self, gmf_model: Model, user_train,  item_train, y_train, eval_data):
        print('-------- Training GMF --------')
        user_eval, item_eval, rating_eval = eval_data
        history = gmf_model.fit([user_train, item_train], 
                      y_train,
                      epochs=self.config['GMF']['epochs'], 
                      validation_data=([user_eval, item_eval], rating_eval),
                      batch_size=self.config['GMF']['batch_size'],
                      verbose=1)
        gmf_model.save(self.config['GMF']['model_parameters_path'])