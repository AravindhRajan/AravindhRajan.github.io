# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:24:44 2020

@author: Aravindh Rajan
"""

import pandas as pd
import numpy as np
import nltk
import spacy
from spacy.lang.en import English 
from spacy.lang.en.stop_words import STOP_WORDS #sw
from spacy.tokenizer import Tokenizer #tokenizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
import streamlit as st

# Load spacy model
nlp = spacy.load('en_core_web_sm', parser=False, entity=False) 

#importing data
df1 = pd.read_csv(r"F:\NIIT\Python\Py_prac\Plot_similarity\movies.csv")

# creating a single column with description from imdb and wiki
df1['fullplot'] = df1['wiki_plot'].astype(str) + "\n" + \
                 df1['imdb_plot'].astype(str)
                 

#test['tweet_without_stopwords'] = test['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df1['plot_filt_lem'] = df1['new_plot'].apply(lambda x: remove_mysw_sp(x))


# custom sw list 
custom_sw = ['justtrying']

# Mark them as stop words
for w in custom_sw:
    nlp.vocab[w].is_stop = True


# new column with filtered, lemmatized words
df1['plot_filt_lem'] = df1.fullplot.apply(lambda text: " ".join(token.lemma_ for token in nlp(text) 
                                                   if not token.is_stop))

# create a tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True,ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in df1['plot_filt_lem']])

# creating similarity matrix from cosing similarity
similarity_distance = 1 - cosine_similarity(tfidf_matrix)


st.write("""
# Natural Language Processing
### This application helps you with a list of movies similar to the one that you choose.
""")

#################################
# asking for user choice
movie_list = df1['title'].to_list()
movie_list2 = ['Select']+movie_list
my_movie = st.selectbox('Choose any movie',movie_list2)

if my_movie:
    n = st.number_input('How many movies do you want',min_value=2,max_value=6)
    # find similar titles
    def suggest_sim_movies(my_movie,n):
        index = df1[df1['title'] == my_movie].index[0]
        vector = similarity_distance[index, :]
        most_similar = df1.iloc[np.argsort(vector)[:n+1], 1].to_list()
        most_similar_list = most_similar[1:n+1]
        return most_similar_list
    
    final_lst = suggest_sim_movies(my_movie,n)
    st.write('Movies similar to your selection:')
    st.write(final_lst)# prints "All Quiet on the Western Front"