# -*- coding: utf-8 -*-
"""
Myers–Briggs Type Indicator classification based on Twitter data

Preprocessing
"""
import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords 
import pickle

# load
print('** load')
all_df=pd.read_csv('mbti_1.csv')

df=all_df.loc[:,'posts'].to_frame()
labels_df=all_df.loc[:,'type'].to_frame()

def category_encoder(to_translate, inverse=False):
    ptypes=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',
            'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',
            'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
    if(inverse):
        return ptypes[to_translate]
    else:
         return ptypes.index(to_translate)

labels_df['category']=[category_encoder(label) for label in labels_df['type']]

print('** features extraction')
posts=[tweets.lower() for tweets in df.posts.tolist()]
feature_labels=['a_url','a_utag']
urlmarked = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))','#URL#',tweets) for tweets in posts]
df['a_url']=[tweets.count('#URL#')/50 for tweets in urlmarked]
url_removed = [tweets.replace('#URL#',' ') for tweets in urlmarked]

usertagmarked= [re.sub('@[^\s]+','#UTG#',tweets) for tweets in url_removed]           
df['a_utag']=[tweets.count('#UTG#')/50 for tweets in usertagmarked]
tags_removed = [tweets.replace('#UTG#',' ') for tweets in usertagmarked]
df['a_ellipsis']=[tweets.count('...')/50 for tweets in tags_removed]
punctuation = '!#$%&\()*+,-./:;<=>?[\\]^_`{}~'+"'"
for pchar in punctuation:
    df['a_'+pchar]=[tweets.count(pchar)/50 for tweets in tags_removed]
    tags_removed=[tweets.replace(pchar,' ') for tweets in tags_removed]
split_tweets=[tweets.split('|||') for tweets in tags_removed]
tokenized_tweets=[[tweets.split() for tweets in tweetsarray]\
                   for tweetsarray in split_tweets]

print('** part of speech')
tokenized_pos=[[nltk.pos_tag(tk_tweet) for tk_tweet in tk_tweets] for tk_tweets in tokenized_tweets]
only_pos=[' '.join([' '.join([postuple[1] for postuple in tk_tweet]) for tk_tweet in tk_tweets]) for tk_tweets in tokenized_pos]

pos_tags=['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
          'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP',
          'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
          'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
for ptag in pos_tags:
    df['a_pos_'+ptag]=[tweets.count('ptag')/50 for tweets in only_pos]

print('** word count')
countwords=[[len(tweetsplit) for tweetsplit in tweetsarray]\
             for tweetsarray in tokenized_tweets]
df['a_countwords']=[np.mean(wordcounts) for wordcounts in countwords]
df['std_countwords']=[np.std(wordcounts) for wordcounts in countwords]
df['med_countwords']=[np.median(wordcounts) for wordcounts in countwords]

print('** stopwords')
sw_removed=tags_removed.copy()
for sword in stopwords.words('english'):
    df['a_sw_'+sword]=[tweets.count(' '+sword+' ')/50 for tweets in tags_removed]
    sw_removed=[tweets.replace(' '+sword+' ',' ') for tweets in sw_removed]

remove_excess=[re.sub('[\s]+', ' ', tweets) for tweets in sw_removed]
clean_tweets=[tweets.replace('|||',' ') for tweets in remove_excess]

print('** sentiment')
sentiment_features=[[dict([(word, True) for word in tweet.split()]) for tweet in tweets]\
                     for tweets in [tweets.split('|||') for tweets in remove_excess]]

NB_nltk_clf,MaxEnt_nltk_clf=pickle.load(open('sentiment_classifiers','rb'))

df['sentiment_nm']=[sum(NB_nltk_clf.classify_many(sentiment_features[i]))/50\
                       for i in range(len(sentiment_features))]
df['sentiment_me']=[sum(MaxEnt_nltk_clf.classify_many(sentiment_features[i]))/50\
                      for i in range(len(sentiment_features))]

df['clean_tweets']=[tweets.replace('|||',' ') for tweets in remove_excess]

# save
df.to_feather('preprocessed_features')
labels_df.to_feather('labels')

#eof