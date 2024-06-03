# Neural network-based Collaborative Filtering with Jester Dataset - Recommendation System

## Overview

This project implements a recommendation system using the Neural Collaborative Filtering (NCF) approach on the Jester dataset. The system is designed to recommend jokes to users based on their preferences.

## Dataset

### Jester Dataset

The Jester dataset, developed by Ken Goldberg and his team at UC Berkeley, is used in this project. The Jester system delivers jokes via an HTML client interface. Users rate these jokes, and upon completing all ratings in the gauge set, the system recommends new jokes based on user preferences.

**Dataset Details:**
- **Ratings Range**: -10.00 to +10.00, indicating user preferences.
- **Duration**: Data collected from April 1999 to May 2003.
- **User Engagement**: Over 4.1 million ratings from 73,421 users.
- **Files Included**:
  - **HTML Joke Files**:
    - Total: 100 files named from `init1.html` to `init100.html`.
    - Each file corresponds to a joke ID used in the Excel files.
  - **Excel Files for Ratings**:
    - `ratings_1.xlsx`: Data from 24,983 users, dimensions 24983 X 101.
    - `ratings_2.xlsx`: Data from 23,500 users, dimensions 23500 X 101.
    - `ratings_3.xlsx`: Data from 24,938 users, dimensions 24938 X 101.

### Data Structure

- **Ratings Data**:
  - Each rating is a real value from -10.00 to +10.00.
  - The placeholder value "99" indicates a joke that has not been rated ("null").
  - Each row in the Excel files represents a user's ratings:
    - The first column shows the count of jokes rated by the user.
    - The subsequent 100 columns contain ratings for jokes ID 01 through ID 100.

### Directory Description

- `data`: Contains the original datasets and the preprocessed datasets.
- `notebooks`: Contains Jupyter notebooks for data preprocessing and dataset building.
  - `preprocess.ipynb`: Notebook for data preprocessing.
  - `build_dataset.ipynb`: Notebook for building the dataset structure.
- `models`: Contains the implementation of different models used in NCF.
  - `GMF.py`: Implementation of Generalized Matrix Factorization.
  - `MLP.py`: Implementation of Multi-Layer Perceptron.
  - `NeuMF.py`: Implementation of Neural Matrix Factorization.
- `neumf-config.yaml`: Configuration file for the NeuMF model.
- `app.py`: Implementation for UI.
- `recommend.py`: Methods for recommending to users.
- `train.ipynb`: Notebook for training the recommendation model.
- `utils`: Extra methods.

## Usage
  to do
## Configuration

The model configuration is stored in `neumf-config.yaml`. You can modify this file to change the model parameters.

### Short Example Configuration for Continous Rating Task(`neumf-config.yaml`)

```yaml
NeuMF:
  epochs: 20
  batch_size: 1024
  learning_rate: 0.001
  loss: "mean_squared_error"
  output_layer_activation: 'linear'
  optimizer: "sgd"
  metrics: ["mae"]

  MLP:
    epochs: 20
    batch_size: 256
    learning_rate: 0.001
    optimizer: "adam"
    loss: "mean_squared_error"
    output_layer_activation: 'linear'
    metrics: ["mae"]
    layers: [64, 32, 16] 
    num_layers: 3
    embedding_out_dim: 32 
    model_parameters_path: "./models/weights/mlp.h5"
    
  GMF:
    epochs: 20
    batch_size: 256
    learning_rate: 0.001
    optimizer: "adam"
    loss: "mean_squared_error"
    output_layer_activation: 'linear'
    metrics: ["mae"]
    factors: 16 
    model_parameters_path: "./models/weights/gmf.h5"
```

## Preprocessing Steps

1. **Cleaning Jokes:**
   - Removed HTML tags and unnecessary characters from the jokes to ensure the text is clean and suitable for analysis.

3. **Preprocessing Ratings:**
   - Transformed the data from a wide format (one row per user with multiple joke ratings) to a long format (one row per joke rating) to facilitate analysis.
   - Sorted the data by user and joke ID for consistency.
   - Handled missing values by replacing the placeholder value "99" (indicating unrated jokes) with NaN and then dropping these rows.

4. **Normalizing Ratings:**
   - Normalized the ratings to a range between 0 and 1. The original ratings ranged from -10 to +10. The normalization was done using the formula: `(rating + 10) / 20`.

## Reference

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. In *Proceedings of the 26th international conference on world wide web* (pp. 173-182). Available at [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031).


