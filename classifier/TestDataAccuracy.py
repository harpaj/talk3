from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np

from textblob import TextBlob
from featureAnalyzer import feat_analyser as sn

import sklearn

data_dir = '../data'

df_train_set = pd.read_csv(data_dir + '/final_train_set.csv', encoding='iso-8859-1')

df_train_set = pd.DataFrame(df_train_set)

df_test_set = pd.read_csv(data_dir + '/final_test_set.csv', encoding='iso-8859-1')

# 1.Pos-Neg Word Count Feature
# For Training Data
df_train_feat1 = sn.defsentfeatextractor(df_train_set)
df_train_set = df_train_set.filter(['post_id', 'sentence', 'treatments', 'sentiment', 'factual'], axis=1)

# For Testing Data
df_test_feat1 = sn.defsentfeatextractor(df_test_set)
df_test_set = df_test_set.filter(['post_id', 'sentence', 'treatments', 'sentiment', 'factual'], axis=1)

"""
#2. TF-IDF
#Train Data

vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(df_train_set["sentence"].map(str)).toarray()
tfidf_train_df = pd.DataFrame(tfidf, columns=[("tfidf_" + str(i)) for i in range(len(tfidf[0]))])
df_train_set = pd.concat((df_train_set, tfidf_train_df), axis=1)

#print(vectorizer.vocabulary)
print(tfidf)
#print(df)

#Test Data
tfidf2 = vectorizer.transform(df_test_set["sentence"].map(str)).toarray()
tfidf_test_df = pd.DataFrame(tfidf2, columns=[("tfidf_" + str(i)) for i in range(len(tfidf2[0]))])
df_test_set = pd.concat((df_test_set, tfidf_test_df), axis=1)
"""

# 3. Polarity Score
# For Training Data
df_train_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_train_set["sentence"].map(str)))
# For Test Data
df_test_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_test_set["sentence"].map(str)))

# 4. Subjectivity Score
# For Training Data

df_train_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_train_set["sentence"].map(str)))
# For Test Data
df_test_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_test_set["sentence"].map(str)))

# 5. Personal Pronoun counts
# Train Data
df_pron_count_train = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_train_set["sentence"].map(str)),
                                   columns=['first_per_pron', 'sec_per_pron', 'Third_per_pron', 'personal_pron',
                                            'total_pron', 'sing_proper_noun'])
df_train_set = pd.concat((df_train_set, df_pron_count_train), axis=1)

# For New Data
df_pron_count_final_set = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_test_set['sentence'].map(str)),
                                       columns=['first_per_pron', 'sec_per_pron', 'Third_per_pron', 'personal_pron',
                                                'total_pron', 'sing_proper_noun'])
df_test_set = pd.concat((df_test_set, df_pron_count_final_set), axis=1)

# 6. Trivial Score
# Train Data
df_train_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_train_set["sentence"].map(str)))
df_test_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_test_set["sentence"].map(str)))

# Merging Feature
# Training Data
df_train_set = pd.concat((df_train_set, df_train_feat1), axis=1)

# Test Data
df_test_set = pd.concat((df_test_set, df_test_feat1), axis=1)

# Features : Only 8 selected after analysing all features
# Features = [ 'first_per_pron', 'sec_per_pron',  'personal_pron', 'total_pron', 'sing_proper_noun', 'trivial_score',  'positive_word_count', 'vander_score']

features = df_train_set.columns[5:]
list_feat = list(features)
list_feat.remove('negative_word_count')
list_feat.remove('Third_per_pron')
list_feat.remove('polarity')
list_feat.remove('subjectivity')
temp_df = pd.DataFrame(columns=list_feat)
features = temp_df.columns

print(features)

# Encoding Target Classes
y, y_index = pd.factorize(df_train_set['sentiment'])
y2, y2_index = pd.factorize(df_train_set['factual'])

# Target Classes
Y = np.column_stack((y, y2))

# Train The Random Forest Classifier
dict_cls_weight = 'balanced'

clf = RandomForestClassifier(n_jobs=1, class_weight=dict_cls_weight, random_state=56)

sample_wt = sklearn.utils.class_weight.compute_sample_weight(dict_cls_weight, Y)

clf.fit(df_train_set[features], Y, sample_weight=sample_wt)

# Applying the classifier on test data
predict_result = clf.predict(df_test_set[features])

# Encoded value for sentiment prediction
A = np.array(predict_result)
B = np.asmatrix(A)
C = B[:, 0]
C = C.astype(int)

C = C.ravel()
C = np.asarray(C)
C = C[0]

# Encoded value for factuality prediction
B = np.asmatrix(A)
C2 = B[:, 1]
C2 = C2.astype(int)

C2 = C2.ravel()
C2 = np.asarray(C2)
C2 = C2[0]

# predicted values for sentiment
y_pred = []
for i in range(len(df_test_set)):
    # print(y_index[C[i]])
    y_pred.append(y_index[C[i]])

y_pred = np.asarray(y_pred)

# predicted values for factual
y2_pred = []
for i in range(len(df_test_set)):
    # print(y2_index[C2[i]])
    y2_pred.append(y2_index[C2[i]])

y2_pred = np.asarray(y2_pred)

# Confusion matrix for sentiment
confusion_matrix = pd.crosstab(df_test_set['sentiment'], y_pred, rownames=['Actual Sentiment'],
                               colnames=['Predicted Sentiment'], margins=True)
print(confusion_matrix)
if 'neu' not in confusion_matrix.columns:
    # print("problem")
    neu = 0
else:
    neu = confusion_matrix["neu"]["neu"]

if 'neg' not in confusion_matrix.columns:
    # print("problem")
    neg = 0
else:
    neg = confusion_matrix["neg"]["neg"]

if 'pos' not in confusion_matrix.columns:
    # print("problem")
    pos = 0
else:
    pos = confusion_matrix["pos"]["pos"]

sentiment_accuracy = (neg + neu + pos) / len(df_test_set)
print("Sentiment Accuracy", sentiment_accuracy)

# Create confusion matrix for factuality
confusion_matrix = pd.crosstab(df_test_set['factual'], y2_pred, rownames=['Actual Factuality'],
                               colnames=['Predicted Factuality'], margins=True)
print(confusion_matrix)
if 'no' not in confusion_matrix.columns:
    # print("problem")
    no = 0
else:
    no = confusion_matrix["no"]["no"]

if 'yes' not in confusion_matrix.columns:
    # print("problem")
    yes = 0
else:
    yes = confusion_matrix["yes"]["yes"]

factual_accuracy = (no + yes) / len(df_test_set)

print("Factual Accuracy", factual_accuracy)
