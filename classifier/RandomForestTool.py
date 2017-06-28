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
from featureAnalyzer import feat_analyser as sn
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#end of method
data_dir = '../data'


print(csv.field_size_limit())
# Create a dataframe with the four feature variables
df_train_set = pd.read_csv(data_dir + '/dataset_train.csv')
print("File dataset", df_train_set.head())
df_train_set = pd.DataFrame(df_train_set)

df_test_set = pd.read_csv(data_dir +'/dataset_test.csv')
#original_col = df_train_set.columns;
#print(original_col)


#1.Pos-Neg Word Count Feature
#For Training Data
df_train_feat1 = sn.defsentfeatextractor(df_train_set)
df_train_set = df_train_set.filter(['post_id', 'sentence', 'after', 'third', 'treatments', 'sentiment', 'factual'], axis=1)


#For Testing Data
df_test_feat1 = sn.defsentfeatextractor(df_test_set)
df_test_set = df_test_set.filter(['post_id', 'sentence', 'after', 'third', 'treatments', 'sentiment', 'factual'], axis=1)



#2. TF-IDF
#Train Data
vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(df_train_set["sentence"].map(str) + "'\n'" + df_train_set["after"].map(str) + "'\n'" + df_train_set["third"].map(str)).toarray()
tfidf_train_df = pd.DataFrame(tfidf, columns=[("tfidf_" + str(i)) for i in range(len(tfidf[0]))])
df_train_set = pd.concat((df_train_set, tfidf_train_df), axis=1)

#print(vectorizer.vocabulary)
print(tfidf)
#print(df)

#Test Data
tfidf2 = vectorizer.transform(df_test_set["sentence"].map(str) + "'\n'" + df_test_set["after"].map(str) + "'\n'" + df_test_set["third"].map(str)).toarray()
tfidf_test_df = pd.DataFrame(tfidf2, columns=[("tfidf_" + str(i)) for i in range(len(tfidf2[0]))])
df_test_set = pd.concat((df_test_set, tfidf_test_df), axis=1)



#3. Polarity Score
#For Training Data
df_train_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_train_set["sentence"].map(str) + "'\n'" + df_train_set["after"].map(str) + "'\n'" + df_train_set["third"].map(str)))
#For Test Data
df_test_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_test_set["sentence"].map(str) + "'\n'" + df_test_set["after"].map(str) + "'\n'" + df_test_set["third"].map(str)))

#4. Subjectivity Score
#For Training Data
df_train_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_train_set["sentence"].map(str) + "'\n'" + df_train_set["after"].map(str) + "'\n'" + df_train_set["third"].map(str)))
#For Test Data
df_test_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_test_set["sentence"].map(str) + "'\n'" + df_test_set["after"].map(str) + "'\n'" + df_test_set["third"].map(str)))


#Merging Feature
#Training Data
df_train_set = pd.concat((df_train_set, df_train_feat1), axis=1)

#Test Data
df_test_set = pd.concat((df_test_set, df_test_feat1), axis=1)


print("concatenated train data:", df_train_set.head())
print("concatenated test data:", df_test_set.head())

#Preprocess Data
# Create a list of the feature column's names
features = df_train_set.columns[8:]

print("Features",features)

#Creating Training and Test Data
#df_train_set['is_train'] = np.random.uniform(0, 1, len(df_train_set)) <= .75
#print(df['is_train'])

# Create two new dataframes, one with the training rows, one with the test rows
#train, test = df_train_set[df_train_set['is_train'] == True], df_train_set[df_train_set['is_train'] == False]


#print(train.head())
#print(test.head())


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(df_train_set))
print('Number of observations in the test data:',len(df_test_set))



# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y , y_index  = pd.factorize(df_train_set['sentiment'])
y2 , y2_index= pd.factorize(df_train_set['factual'])
print(y)
print(y_index)
print("printing Y2")
print(y2)
print(y2_index)
print(len(y))
print(len(y2))


Y=np.column_stack((y,y2))
dict_cls_weight = [{0:1,1:3,2:3},{0:3,1:3}]
#Train The Random Forest Classifier
# Create a random forest classifier. By convention, clf means 'classifier'

sample_wt = sklearn.utils.class_weight.compute_sample_weight(dict_cls_weight,Y)
clf = RandomForestClassifier(n_jobs=2,class_weight=dict_cls_weight)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
print(df_train_set[features])
print(df_test_set[features])


clf.fit(df_train_set[features], Y,sample_weight=sample_wt)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
predict_result =clf.predict(df_test_set[features])

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
#predict_result_prob=clf.predict_proba(df_test_set[features])[0:len(df_test_set)]
#print(predict_result_prob)
#--------------------------------------------------------------------

#Evaluate classifier
#predicted values for sentiment
y_pred = []
for i in range(len(df_test_set)):
    print(y_index[C[i]])
    y_pred.append(y_index[C[i]])
#print(preds2)
y_pred =np.asarray(y_pred)
print(y_pred)
print("Test ",df_test_set['sentiment'])


#predicted values for factual
y2_pred = []
for i in range(len(df_test_set)):
    print(y2_index[C2[i]])
    y2_pred.append(y2_index[C2[i]])
#print(preds2)
y2_pred =np.asarray(y2_pred)
print(y2_pred)
print("Test ",df_test_set['factual'])


# Create confusion matrix
confusion_matrix=pd.crosstab(df_test_set['sentiment'], y_pred, rownames=['Actual Sentiment'], colnames=['Predicted Sentiment'],margins=True)
print(confusion_matrix)

#Precision
print("Precision: ",precision_score(df_test_set['sentiment'], y_pred,average=None))
print("Precision avg=micro: ",precision_score(df_test_set['sentiment'], y_pred,average='macro'))
print("Precision avg=weighted: ",precision_score(df_test_set['sentiment'], y_pred,average='weighted'))
#Recall

print("Precision: ",recall_score(df_test_set['sentiment'], y_pred,average=None))
print("Precision avg=micro: ",recall_score(df_test_set['sentiment'], y_pred,average='macro'))
print("Precision avg=weighted: ",recall_score(df_test_set['sentiment'], y_pred,average='weighted'))



# Create confusion matrix
confusion_matrix=pd.crosstab(df_test_set['factual'], y2_pred, rownames=['Actual Factuality'], colnames=['Predicted Factuality'],margins=True)
print(confusion_matrix)

#Precision
print("Precision: ",precision_score(df_test_set['factual'], y2_pred,average=None))
print("Precision avg=micro: ",precision_score(df_test_set['factual'], y2_pred,average='macro'))
print("Precision avg=weighted: ",precision_score(df_test_set['factual'], y2_pred,average='weighted'))

#Recall
print("Precision: ",recall_score(df_test_set['factual'], y2_pred,average=None))
print("Precision avg=micro: ",recall_score(df_test_set['factual'], y2_pred,average='macro'))
print("Precision avg=weighted: ",recall_score(df_test_set['factual'], y2_pred,average='weighted'))
