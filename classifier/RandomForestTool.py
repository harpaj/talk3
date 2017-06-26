# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

import nltk
import string

# Load numpy
import numpy as np
import  csv
from nltk.corpus import stopwords
from textblob import TextBlob

def defsentfeatextractor(dataframe):
    lexicon_directory = 'C:/Users/Atin/Documents/GitHub/New folder/talk3/data/opinion-lexicon-English'
    data_dir = 'C:/Users/Atin/Documents/GitHub/New folder/talk3/data'
    stop = stopwords.words('english')

    data_review = pd.DataFrame(dataframe)

    # print(data_review.head())
    # print(data_review.describe())


    # Positive Lexicons\n
    positive_words = np.loadtxt(lexicon_directory + '/positive-words.txt', comments=';', dtype='bytes')
    positive_words = [x.decode('us-ascii') for x in positive_words]
    positive_words = set(positive_words)
    # print(positive_words)

    # stop words\n
    stop_words = np.loadtxt(data_dir + '/stopwords.txt', comments=';', dtype='bytes')
    stop_words = [x.decode('us-ascii') for x in stop_words]
    stop_words = set(stop_words)
    # print(stop_words)

    # Negative Lexicons
    with open(lexicon_directory + '/negative-words.txt', encoding='iso-8859-1') as f:
        negative_words = np.loadtxt(f, comments=';', dtype='bytes')
        negative_words = [x.decode('iso-8859-1') for x in negative_words.tolist()]
        negative_words = set(negative_words)
    # print(negative_words)



    # Get only the required data
    # data_review = data_review[['post_id','agrees', 'before', 'sentence', 'after', 'third', 'treatments','factual (yes/no)','sentiment (pos/neg/neu)']]
    # print(data_review.head())
    pd.set_option('display.width', 1000)

    # merge coloumns before , sentence,after and third
    df = pd.DataFrame(data_review)
    df["text"] = df["sentence"].map(str) + "'\n'" + df["after"].map(str) + "'\n'" + df["third"].map(str)

    def remove_punctuation(s):
        s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
        return s

    df['cleaned'] = df['text'].apply(remove_punctuation)
    df['cleaned'] = [token.lower() for token in df['cleaned']]

    pd.options.display.max_colwidth = 100

    # nltk.word_tokenize(df.get_value(0,'text'))
    data_review['tokens'] = df['cleaned'].apply(nltk.word_tokenize)
    data_review['tokens'].apply(lambda x: [item for item in x if item not in stop])

    # Count the number of tokens\
    data_review['word_count'] = data_review['tokens'].apply(len)

    # positive words

    # print(positive_words.intersection(data_review.get_value(0, 'tokens')))
    data_review['positive_word_count'] = data_review['tokens'].apply(
        lambda tokens: len(positive_words.intersection(tokens)))

    # print(data_review['positive_word_count'] )

    data_review['negative_word_count'] = data_review['tokens'].apply(
        lambda tokens: len(negative_words.intersection(tokens)))

    data_review = data_review[['negative_word_count', 'positive_word_count', 'tokens', 'factual', 'sentiment']]
    # print(data_review[['negative_word_count','positive_word_count'] ])

    return data_review[['negative_word_count', 'positive_word_count']]


#end of method


print(csv.field_size_limit())
# Create a dataframe with the four feature variables
df_set = pd.read_csv('D:/Projects/TALK-3/Annotation/Atin/dataset_trial.csv')
print("File dataet",df_set.head())
df_set = pd.DataFrame(df_set)
original_col = df_set.columns;
print(original_col)

print("File dataet after",df_set.head())

#1.Pos-Neg Word Count Feature

df_feat1 = defsentfeatextractor(df_set)
df_set = df_set.filter(['post_id','sentence','after','third','treatments','sentiment','factual'], axis=1)
#print("new",df_set.head())

print("SentFeat:",df_feat1.head())
#print("Dataset",df_set.head())

#print(df_set.columns)

#2. TF-IDF


#3. Polarity Score
df_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_set["sentence"].map(str)+ "'\n'" + df_set["after"].map(str)+ "'\n'" +df_set["third"].map(str)))

#4. Subjectivity Score
df_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_set["sentence"].map(str)+ "'\n'" + df_set["after"].map(str)+ "'\n'" +df_set["third"].map(str)))

print(df_set.head())
#Merging Feature
df_set = pd.concat((df_set, df_feat1), axis=1)


print("concatenated DF:",df_set.head())
#Preprocess Data
# Create a list of the feature column's names
features = df_set.columns[7:]

print("Features",features)

#Creating Training and Test Data
df_set['is_train'] = np.random.uniform(0, 1, len(df_set)) <= .75
#print(df['is_train'])

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df_set[df_set['is_train']==True], df_set[df_set['is_train']==False]


#print(train.head())
#print(test.head())


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))



# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y , y_index  = pd.factorize(train['sentiment'])
y2 , y2_index= pd.factorize(train['factual'])
print(y)
print(y_index)
print("printing Y2")
print(y2)
print(y2_index)
print(len(y))
print(len(y2))


Y=np.column_stack((y,y2))

#Train The Random Forest Classifier
# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=2)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
print(train[features])
print(test[features])
clf.fit(train[features], Y)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
predict_result =clf.predict(test[features])

print(predict_result)

#encoded value for sentiment
A = np.array(predict_result)
B=np.asmatrix(A)
C=B[:,0]
C=C.astype(int)
print(C)


C=C.ravel()
C = np.asarray(C)
C=C[0]
print(C)

#encoded value for factuality
B=np.asmatrix(A)
C2=B[:,1]
C2=C2.astype(int)
print(C2)


C2=C2.ravel()
C2 = np.asarray(C2)
C2=C2[0]
print(C2)

#------------------------------------------------------------------
# View the predicted probabilities of the first 10 observations
predict_result_prob=clf.predict_proba(test[features])[0:len(test)]
print(predict_result_prob)
#--------------------------------------------------------------------

#Evaluate classifier
#predicted values for sentiment
y_pred = []
for i in range(len(test)):
    print(y_index[C[i]])
    y_pred.append(y_index[C[i]])
#print(preds2)
y_pred =np.asarray(y_pred)
print(y_pred)
print("Test ",test['sentiment'])


#predicted values for factual
y2_pred = []
for i in range(len(test)):
    print(y2_index[C2[i]])
    y2_pred.append(y2_index[C2[i]])
#print(preds2)
y2_pred =np.asarray(y2_pred)
print(y2_pred)
print("Test ",test['factual'])


# Create confusion matrix
confusion_matrix=pd.crosstab(test['sentiment'], y_pred, rownames=['Actual Sentiment'], colnames=['Predicted Sentiment'],margins=True)
print(confusion_matrix)


# Create confusion matrix
confusion_matrix=pd.crosstab(test['factual'], y2_pred, rownames=['Actual Factuality'], colnames=['Predicted Factuality'],margins=True)
print(confusion_matrix)
