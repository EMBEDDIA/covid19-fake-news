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
| sentence_bert | [distilbert-base-nli-mean-tokens](./src/sentence_bert/sentence_transformers/README.md) | 0.949 |  0.912 |
| sentence_bert | [roberta-large-nli-stsb-mean-tokens](./src/sentence_bert/sentence_transformers/README.md) | 0.953 |  0.919 |
| sentence_bert | [xlm-roberta-base](./src/sentence_bert/sentence_transformers/README.md) | 0.959 |  0.906 |
| linear_SVM | [Tax2Vec-+-tf-idf](./src/tax2vec/README.md) |     | 0.949 |  
| distilBERT | [distilBERT-tokenizer](./src/distilBERT/README.md) | 0.978 |  |  
