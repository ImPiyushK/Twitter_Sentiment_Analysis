# Twitter Sentiment Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization.
![Display Image](./Rawdata/cap3.jpeg)

## Introduction
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, lets say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

The data has been downloaded from [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/).

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, your objective is to predict the labels on the given test dataset. Since it is a supervised learning task we are provided with a training data set which consists of Tweets labeled with “1” or “0” and a test data set without labels.
* label “0”: Positive Sentiment
* label “1”: Negative Sentiment

## Workflow
![](./Rawdata/cap4.png)

## Exploratory Data Analysis
![](./Rawdata/cap1.png)  ![](./Rawdata/cap2.png)

The above two graphs show that the given data is an imbalanced one with very less amount of “1” labels and the length of the tweet doesn’t play a major role in classification.

## Data preprocessing and Feature Engineering
The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

The given data sets are comprised of very much unstructured tweets which should be preprocessed to make an NLP model. In this project, I've tried out the following techniques of preprocessing the raw data. But the preprocessing techniques is not limited.
* Removal of punctuations.
* Removal of commonly used words (stopwords).
* Normalization of words.

## Model Selection
Before I let our data to train I have to numerically represent the preprocessed data. So, I have vectorized sting data to numerical values using Count Vectorization and Tf-Idf in order to feed it to a machine learning algorithm. I've choose naive bayes classifier for this binary classification since it is the most common algorithm used in NLP.

## Model Validation
Accuracy is measured using the built-in function of scikit-learn, confusion matrix and classification report.
An accuracy of 0.93962 is obtained for the pipelined model of Count Vectorization, Tf-Idf and Naive Bayes.

## Improvement
Accuracy can also be improved by tuning parameters using GridSearchCV and other preprocessing techniques.
