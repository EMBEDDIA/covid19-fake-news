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


## Results Train Dev Test split.

| Model name   |      Vectorization      |  Train score | Test score |  Validation Score |
|----------|:-------------:|:------:|:------:|:------:|
| TFIDF_LSA |  [LSA](./src/lsa_baseline/README.md) | 0.92 (avg 10-fold CV) | 0.77 | 0.93 |
| distilBERT | [distilBERT-tokenizer](./src/distilBERT/README.md) |  | 0.97 | 0.98 | 
| linear_SVM | [Tax2Vec-+-tf-idf](./src/tax2vec/README.md) |    | 0.92 | 0.94 |
| linear_SVM | [Tax2Vec(knowledge graph)-+-tf-idf](./src/tax2vec_knowledge_graphs/README.md) |    | 0.93 | 0.93 |
| gradient_boosting | [Tax2Vec(knowledge graph)-+-tf-idf-+-distilBERT_tokenization](.) |    | 0.92 | 0.94 |

## Results Train Dev Test split (EVALUATION BK)

| Model name   |      Vectorization      |   Train tactic | Train score | DEV score |  Test Score |
|----------|:-------------:|:------:|:------:|:------:|:------:|
| TFIDF_LSA |  [LSA](./src/lsa_baseline/README.md) | (avg 10-fold CV) |  0.9658 | 0.9302 | 0.9281 |
| Stloymetric features | [handcrafted-features](./src/statistical_baseline/README.md) | (avg 10-fold CV) | 0.7861 | 0.7903 | 0.7805 |
| sentence_bert | [distilbert-base-nli-mean-tokens](./src/sentence_bert/sentence_transformers/README.md) |   (avg 10-fold CV) | 0.9365 | 0.9124 | 0.9113 |
| sentence_bert | [roberta-large-nli-stsb-mean-tokens](./src/sentence_bert/sentence_transformers/README.md) |   (avg 10-fold CV) | 0.9623 | 0.9184 | 0.9142 |
| sentence_bert | [xlm-r-large-en-ko-nli-ststb](./src/sentence_bert/sentence_transformers/README.md) |   (avg 10-fold CV) |0.9376 | 0.9226 | 0.9124 |
| distilBERT | [distilBERT-tokenizer](./src/distilBERT/README.md) |  x  | 0.9933 | 0.9807 | 0.9708
| stacking_probs | [lsa_sentence_bert_stlyometrics](./src/sentence_bert/stacking) |  NN | 0.9710 | 0.9380 | 0.9390 |
| stacking | [lsa_sentence_bert_stlyometrics](./src/sentence_bert/stacking) | (10-fold CV) | 0.9695 | 0.9445 | 0.9425 |
| linear_SVM | [Tax2Vec-+-tf-idf](./src/tax2vec/README.md) |  |  | 0.94 | 0.92 |
| linear_SVM | [Tax2Vec(knowledge graph)-+-tf-idf](./src/tax2vec_knowledge_graphs/README.md) |  |  | 0.93 | 0.93 |
| gradient_boosting | [Tax2Vec(knowledge graph)-+-tf-idf-+-distilBERT_tokenization](.) |  |  | 0.94 | 0.92 |
