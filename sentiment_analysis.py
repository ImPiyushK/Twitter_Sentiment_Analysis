# -*- coding: utf-8 -*-
"""Sentiment Analysis(Piyush).ipynb

# **Twitter Sentiment Analysis**

**Importing Libraries**
"""

#Data Analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

"""**Reading and Extracting data from .csv files**"""

train_tweets = pd.read_csv("/content/drive/My Drive/Sentiment Analysis dataset/train.csv")
test_tweets = pd.read_csv("/content/drive/My Drive/Sentiment Analysis dataset/test.csv")

train_tweets.head()
#test_tweets.head()

train_tweets = train_tweets[['label', 'tweet']]
test = test_tweets['tweet']

"""**Exploratory Data Analysis**"""

train_tweets['length'] = train_tweets['tweet'].apply(len)
fig1 = sns.barplot('label', 'length', data = train_tweets)
plt.title("Average Word Length vs Label")
plot = fig1.get_figure()

fig2 = sns.countplot(x="label", data=train_tweets)
plt.title("Label Counts")
plot = fig2.get_figure()

"""**Data preprocessing and Feature Engineering**"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def text_processing(tweet):
    
    #Remoced hastags and other punctuations removed
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)

    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']  #removes user
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)] #removes anything except string
        clean_s = ' '.join(clean_tokens) #created sentence from words
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')] #removed stop words
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)

    #Normalizing the words in tweets 
    def normalization(tweet_list):  #tweet_list contain words returned by no_user_alpha()
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v') #verb
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    final_process_tweet = normalization(no_punc_tweet)
    
    return final_process_tweet

train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)

train_tweets[train_tweets['label']==1].head()

"""**Vectorization and Model Selection**"""

X = train_tweets['tweet']
y = train_tweets['label']
test = test_tweets['tweet']

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)

"""**Model Validation**"""

predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))