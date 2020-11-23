# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:02:22 2020

@author: Bosec
"""

import numpy as np
import lsa_model 
import sentence_transfomers
import statistical_baseline
text = "ncludes high-quality download in MP3, FLAC and more. Paying supporters also get unlimited streaming via the free Bandcamp app."


def get_features(texts):
    lsa_features = lsa_model.fit(texts)
    bert_features = sentence_transfomers.fit(texts, "distilbert-base-nli-mean-tokens")
    stat_features = statistical_baseline.fit(texts)
    roberta_features = sentence_transfomers.fit(texts, "roberta-large-nli-stsb-mean-tokens")
    xlm_features = sentence_transfomers.fit(texts, "xlm-r-large-en-ko-nli-ststb")    
    return np.hstack(([lsa_features, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    
def get_features_probs(texts):
    lsa_features = lsa_model.fit_probs(texts)
    bert_features = sentence_transfomers.fit_probs(texts, "distilbert-base-nli-mean-tokens")
    stat_features = statistical_baseline.fit_probs(texts)
    roberta_features = sentence_transfomers.fit_probs(texts, "roberta-large-nli-stsb-mean-tokens")
    xlm_features = sentence_transfomers.fit_probs(texts, "xlm-r-large-en-ko-nli-ststb")    
    return np.hstack(([lsa_features, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    
print(get_features([text]))
print(get_features_probs([text]))

