import numpy as np
import pandas as pd
import nltk
import datetime
import matplotlib.pyplot as plt
import math
import random as random


lexicon_directory = '../data/opinion-lexicon-English'
data_dir = '../data'


data_review = pd.read_csv(data_dir + '/reannotation.csv')

print(data_review.head())
print(data_review.describe())


# Positive Lexicons\n
positive_words = np.loadtxt(lexicon_directory + '/positive-words.txt', comments=';', dtype='bytes')
positive_words = [x.decode('us-ascii') for x in positive_words]
positive_words = set(positive_words)
print(positive_words)

# Negative Lexicons
with open(lexicon_directory + '/negative-words.txt', encoding='iso-8859-1') as f:
    negative_words = np.loadtxt(f, comments=';', dtype='bytes')
    negative_words = [x.decode('iso-8859-1') for x in negative_words.tolist()]
    negative_words = set(negative_words)
print(negative_words)



# Get only the required data
data_review = data_review[['post_id','agrees', 'before', 'sentence', 'after', 'third', 'treatments','factual (yes/no)','sentiment (pos/neg/neu)']]
print(data_review.head())
pd.set_option('display.width', 1000)

#merge coloumns before , sentence,after and third
df=pd.DataFrame(data_review)
df["text"] = df["before"].map(str) +df["after"].map(str)+ df["third"].map(str)+ df["sentence"]

print("hello")

print(df.head(2))
pd.options.display.max_colwidth = 100

nltk.word_tokenize(df.get_value(0,'text'))
data_review['tokens'] =df['text'].apply(nltk.word_tokenize)
print("sdsdd")
print(data_review['tokens'].head(2))
#print(data_review['tokens'].head())

#Count the number of tokens\
data_review['word_count'] = data_review['tokens'].apply(len)

#positive words

#print(positive_words.intersection(data_review.get_value(0, 'tokens')))
data_review['positive_word_count'] = data_review['tokens'].apply(lambda tokens: len(positive_words.intersection(tokens)))

print(data_review['positive_word_count'] )

data_review['negative_word_count'] = data_review['tokens'].apply(lambda tokens: len(negative_words.intersection(tokens)))

#print(data_review.head())

#df_a = pd.DataFrame(data_review, columns = ['post_id',])