import numpy as np
import pandas as pd
import nltk
import string
from collections import Counter
import re
import datetime
import matplotlib.pyplot as plt
import math
import random as random
from textblob import TextBlob

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class feat_analyser:
    global lexicon_directory
    lexicon_directory = '../data/opinion-lexicon-English'
    global data_dir
    data_dir = '../data'


    def defsentfeatextractor(dataframe):

        sid = SentimentIntensityAnalyzer()

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
        df["text"] = df["sentence"]
        def remove_punctuation(s):
            s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
            return s

        df['cleaned'] = df['text'].apply(remove_punctuation)
        df['cleaned'] = [token.lower() for token in df['cleaned']]

        pd.options.display.max_colwidth = 100

        #nltk.word_tokenize(df.get_value(0,'text'))
        data_review['tokens'] =df['cleaned'].apply(nltk.word_tokenize)
        data_review['tokens'].apply(lambda x: [item for item in x if item not in stop_words])


        #Count the number of tokens\
        data_review['word_count'] = data_review['tokens'].apply(len)

        #positive words

        #print(positive_words.intersection(data_review.get_value(0, 'tokens')))
        data_review['positive_word_count'] = data_review['tokens'].apply(lambda tokens: len(positive_words.intersection(tokens)))

        #print(data_review['positive_word_count'] )

        data_review['negative_word_count'] = data_review['tokens'].apply(lambda tokens: len(negative_words.intersection(tokens)))

        data_review= data_review[['negative_word_count','positive_word_count','tokens']]
        #print(data_review[['negative_word_count','positive_word_count'] ])


        for index, row in data_review.iterrows():
            # print(row['tokens'])
            cleaned_text = filter(lambda x: x not in stop_words, row['tokens'])
            wordlist = nltk.FreqDist(cleaned_text)
            word_features = wordlist.items()
            #print(word_features)

        #print(data_review.head(20))

        def print_sentiment_scores(sentence):
            sent = ""
            snt = sid.polarity_scores(sentence)
            return snt['compound']

        data_review['vander_score'] = df['text'].apply(print_sentiment_scores)
        #print(data_review.head(20))

        return data_review[['negative_word_count','positive_word_count','vander_score']]





    def get_trivial_score(line):
        trivial_score = 2
        line_sets = line.split('.' or '!' or '?')
        regex_count = 0
        for ln in line_sets:
            print(ln)
            matchObj = re.findall(r'i.*heard', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()
            matchObj = re.findall(r'i.*read', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()
            matchObj = re.findall(r'not.*sure', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()
            matchObj = re.findall(r'referred.*me', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()
            matchObj = re.findall(r'[^I].*mention', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()
            matchObj = re.findall(r'((study|studies).*suggest)', ln, re.IGNORECASE)
            regex_count = regex_count + matchObj.__len__()

        if regex_count > 0:
            trivial_score = trivial_score - 1

        url_count = re.findall(r'(https?://[^\s]+)', line)

        if url_count > 0:
            trivial_score = trivial_score - 1

        if abs(TextBlob(line).polarity) > 0.6:
            trivial_score = trivial_score + 1
        return trivial_score

    def count_pronouns(text):
        tagger = TextBlob(text).tags
        print(tagger)
        pron_list = []
        count_prp = Counter(tag for word, tag in tagger if tag == 'PRP' or tag == 'PRP$')
        count_nnp = Counter(tag for word, tag in tagger if tag == 'NNP')  # singular proper noun
        for tag_pair in tagger:
            if (tag_pair[1] == 'PRP') or (tag_pair[1] == 'PRP$') or (tag_pair[1] == 'WP') or (tag_pair[1] == 'WP$'):
                pron_list.append(tag_pair[0])
        print(count_prp['PRP'])
        print(pron_list)

        first_person_pron_list = ['I', 'me', 'mine', 'my', 'we', 'our', 'ours', 'us']
        sec_person_pron_list = ['you', 'your', 'yours']
        third_person_pron_list = ['he', 'she', 'it', 'his', 'hers', 'him', 'her', 'they', 'them', 'their', 'theirs',
                                  'its']

        pron_list = [item.lower() for item in pron_list]
        first_person_pron_list = [item.lower() for item in first_person_pron_list]
        sec_person_pron_list = [item.lower() for item in sec_person_pron_list]
        third_person_pron_list = [item.lower() for item in third_person_pron_list]

        count_first_per_pron = (set(first_person_pron_list) & set(pron_list)).__len__()
        count_sec_per_pron = (set(sec_person_pron_list) & set(pron_list)).__len__()
        count_third_per_pron = (set(third_person_pron_list) & set(pron_list)).__len__()

        print(set(first_person_pron_list) & set(pron_list))
        print(count_first_per_pron)

        return [count_first_per_pron, count_sec_per_pron, count_third_per_pron, count_prp['PRP'] + count_prp['PRP$'],
                len(pron_list), count_nnp['NNP']]

