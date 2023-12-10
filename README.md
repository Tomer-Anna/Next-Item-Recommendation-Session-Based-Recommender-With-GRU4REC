# Retail Rocket
The dataset consists of three files: a file with behaviour data (events.csv), a file with item properties (itemproperties.сsv) and a file, which describes category tree (categorytree.сsv). The data has been collected from a real-world ecommerce website. It is raw data, i.e. without any content transformations, however, all values are hashed due to confidential issues.

The behaviour data, i.e. events like clicks, add to carts, transactions, represent interactions that were collected over a period of 4.5 months. A visitor can make three types of events, namely “view”, “addtocart” or “transaction”. In total there are 2 756 101 events including 2 664 312 views, 69 332 add to carts and 22 457 transactions produced by 1 407 580 unique visitors. For about 90% of events corresponding properties can be found in the “item_properties.csv” file.

Took direct from [paperswithcode](https://paperswithcode.com/dataset/retailrocket) and [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

in this case im using only the interaction dataset (events.csv)

## Task 
Task: Predict the next product.

In this case, I'm trying to predict the next item in the session without distinguishing between the event types. Even though I split the sequences by user, and this could be a user-item problem, I solve it as an item-item problem. Given a sequence of products, I predict the next product in the session.

## Solution
To solve this task I develop a retrieval models using Two-Tower architecture:
A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.
A candidate model computing the candidate representation (an equally-sized vector) using the candidate features
The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.

Within the framework of the Two-Tower architecture, this project specifically employs the query tower, utilizing a GRU layer to adeptly encode the sequence of historical products. The GRU's inherent ability to capture temporal dependencies proves invaluable in understanding the sequential nature of user interactions with products over time. Simultaneously, the candidate tower remains unchanged, consistently providing a robust representation of candidate products.

Reference: 
- [GRU4Rec paper](https://arxiv.org/abs/1511.06939)
- [TensorFlow Recommenders Framework](https://www.tensorflow.org/recommenders/)


## Data Cleaning 
1. Remove items that occur consecutively in the sequence.
2. Exclude users with minimal and excessively extensive activities.
3. Fix columns name.

## Data Preprocessing
1. Group by user.
2. create a sequences with length of n: [1,2,3,4,5,6,7,8] ---> [[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7], [4,5,6,7,8]]
3. Make the last observation (product) in each sequence as target: seq:[3,4,5,6,7] target:[8] 
4. convert the DataFrame into Tensor Dataset.
5. Split the dataset to train validation and test (70%, 15%, 15%).

## Model Training
Training the model with: 
- window_size: 6
- embedding size: 256 
- batch size: 256
- dropout: 0.25
- learning rate: 0.06258085768442055

## Evaluation
1. Prediction example.
2. Evaluation with relevant metrics.
    - Recall@20 at 20: 0.3
    - MMR@20: 0.17        
    - NDCG@20 at 20: 0.3

    

## The Whole Process With Hyper Parameter Tuning
The whole process in one piece of code including hyper parameter tuning.
