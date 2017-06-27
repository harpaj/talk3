import numpy as np
import pandas as pd
import nltk
import string
import re
import datetime
import matplotlib.pyplot as plt
import math
import random as random

from nltk.corpus import stopwords

class sentfeature:
    global lexicon_directory
    lexicon_directory = '../data/opinion-lexicon-English'
    global data_dir
    data_dir = '../data'


    def defsentfeatextractor(dataframe):
        stop = stopwords.words('english')




        data_review = pd.DataFrame(dataframe)

        #print(data_review.head())
        #print(data_review.describe())


        # Positive Lexicons\n
        positive_words = np.loadtxt(lexicon_directory + '/positive-words.txt', comments=';', dtype='bytes')
        positive_words = [x.decode('us-ascii') for x in positive_words]
        positive_words = set(positive_words)
        #print(positive_words)

         #stop words\n
        stop_words = np.loadtxt(data_dir + '/stopwords.txt', comments=';', dtype='bytes')
        stop_words = [x.decode('us-ascii') for x in stop_words]
        stop_words = set(stop_words)
        #print(stop_words)

        # Negative Lexicons
        with open(lexicon_directory + '/negative-words.txt', encoding='iso-8859-1') as f:
            negative_words = np.loadtxt(f, comments=';', dtype='bytes')
            negative_words = [x.decode('iso-8859-1') for x in negative_words.tolist()]
            negative_words = set(negative_words)
        #print(negative_words)



        # Get only the required data
        #data_review = data_review[['post_id','agrees', 'before', 'sentence', 'after', 'third', 'treatments','factual (yes/no)','sentiment (pos/neg/neu)']]
        #print(data_review.head())
        pd.set_option('display.width', 1000)

        #merge coloumns before , sentence,after and third
        df=pd.DataFrame(data_review)
        df["text"] = df["sentence"].map(str)+ "'\n'" + df["after"].map(str)+ "'\n'" +df["third"].map(str)

        def remove_punctuation(s):
            s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
            return s

        df['cleaned'] = df['text'].apply(remove_punctuation)
        df['cleaned'] = [token.lower() for token in df['cleaned']]

        pd.options.display.max_colwidth = 100

        #nltk.word_tokenize(df.get_value(0,'text'))
        data_review['tokens'] =df['cleaned'].apply(nltk.word_tokenize)
        data_review['tokens'].apply(lambda x: [item for item in x if item not in stop])


        #Count the number of tokens\
        data_review['word_count'] = data_review['tokens'].apply(len)

        #positive words

        #print(positive_words.intersection(data_review.get_value(0, 'tokens')))
        data_review['positive_word_count'] = data_review['tokens'].apply(lambda tokens: len(positive_words.intersection(tokens)))

        #print(data_review['positive_word_count'] )

        data_review['negative_word_count'] = data_review['tokens'].apply(lambda tokens: len(negative_words.intersection(tokens)))

        data_review= data_review[['negative_word_count','positive_word_count','tokens','factual','sentiment']]
        #print(data_review[['negative_word_count','positive_word_count'] ])

        return data_review[['negative_word_count','positive_word_count'] ]





