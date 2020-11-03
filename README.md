# Covid19 Fake News


## Model structure
`` Path to model ``
- src/<model_name>
`` Model module structure ``
### Folders
- imgs - plots and figures
- pickles - model files needed to reproduction, execution externally
- logs - log files if you output any
### Scripts
- config.py - file containing hyperparameters for execution and parameters of pickle files and ... 
- model_name.py - script for exectuiioning of the model

#### Advised structure: 
    - train(X,y, **params) -> function for training models, should output lang model + classifier
    - fit(X,y) -> function for fitting the trained model with new data
    - evaluate(X,y) -> function for elvauation of the model data

## Current results

| Model name   |      Vectorization      |  Train score | Validation Score |
|----------|:-------------:|:------:|------:|
| TFIDF_LSA |  [LSA](./src/lsa_baseline/README.md) | 0.944 (avg 10-fold CV)  | 0.931 |
| Stloymetric features | [handcrafted-features](./src/statistical_baseline/README.md) |    0.787 |0.803 |
| linear_SVM | Tax2Vec + tf-idf |     | 0.949 |  

