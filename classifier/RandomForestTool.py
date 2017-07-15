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


#print(csv.field_size_limit())
# Create a dataframe with the four feature variables
df_train_set = pd.read_csv(data_dir + '/final_train_set.csv',encoding = 'iso-8859-1')
#print("File dataset", df_train_set.head())
df_train_set = pd.DataFrame(df_train_set)

df_test_set = pd.read_csv(data_dir +'/final_test_set.csv',encoding ='iso-8859-1')

df_final_set_original=pd.read_csv(data_dir +'/treatment_detected.csv')

#original_col = df_train_set.columns;
#print(original_col)


#1.Pos-Neg Word Count Feature
#For Training Data
df_train_feat1 = sn.defsentfeatextractor(df_train_set)
df_train_set = df_train_set.filter(['post_id', 'sentence','treatments', 'sentiment', 'factual'], axis=1)


##new data
#extract features into a set
df_final_feat = sn.defsentfeatextractor(df_final_set_original)
print(df_final_feat.head(2))
df_final_set=df_final_set_original.filter(['post_id', 'sentence','treatments'])
print("final",df_final_set.head(10))

#For Testing Data
df_test_feat1 = sn.defsentfeatextractor(df_test_set)
df_test_set = df_test_set.filter(['post_id', 'sentence','treatments', 'sentiment', 'factual'], axis=1)



#2. TF-IDF
#Train Data
"""
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

"""

#3. Polarity Score
#For Training Data
df_train_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_train_set["sentence"].map(str)))
#For Test Data
df_test_set['polarity'] = pd.Series(TextBlob(x).polarity for x in  (df_test_set["sentence"].map(str)))
#commented on 07-07-2017 by Atin
#df_final_set['polarity']=pd.Series((TextBlob(x).polarity for x in (df_test_set["sentence"])))
df_final_set['polarity']=pd.Series((TextBlob(x).polarity for x in (df_final_set["sentence"].map(str))))



#print(df_final_set.head(2))



#4. Subjectivity Score
#For Training Data

df_train_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_train_set["sentence"].map(str)))
#For Test Data
df_test_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_test_set["sentence"].map(str)))
#commented on 07-07-2017 by Atin
#df_final_set['subjectivity']=pd.Series((TextBlob(x).polarity for x in (df_test_set["sentence"])))
df_final_set['subjectivity']=pd.Series((TextBlob(x).polarity for x in (df_final_set["sentence"].map(str))))


#5. Personal Pronoun counts
#Train Data
df_pron_count_train = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_train_set["sentence"].map(str)),columns=['first_per_pron','sec_per_pron','Third_per_pron','personal_pron','total_pron','sing_proper_noun'])
df_train_set = pd.concat((df_train_set, df_pron_count_train), axis=1)

#For New Data
df_pron_count_final_set = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_final_set['sentence'].map(str)),columns=['first_per_pron','sec_per_pron','Third_per_pron','personal_pron','total_pron','sing_proper_noun'])
df_final_set = pd.concat((df_final_set, df_pron_count_final_set), axis=1)

#5. Trivial Score
#Train Data
df_train_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_train_set["sentence"].map(str)))
#For New Data
df_final_set['trivial_score']= pd.Series((sn.get_trivial_score(x) for x in (df_final_set["sentence"].map(str))))

#Merging Feature
#Training Data
df_train_set = pd.concat((df_train_set, df_train_feat1), axis=1)

#Test Data
df_test_set = pd.concat((df_test_set, df_test_feat1), axis=1)

df_final_set =pd.concat((df_final_set,df_final_feat),axis=1)
print("desc",df_final_set.describe())
print(df_final_set.head(2))
#print("concatenated train data:", df_train_set.head())
#print("concatenated test data:", df_test_set.head())


#Preprocess Data
# Create a list of the feature column's names
features = df_train_set.columns[7:]
final_features=pd.DataFrame
#changed on 07-07-2017 by Atin
#final_features=df_final_set[['negative_word_count','positive_word_count','polarity','subjectivity','vander_score']]
final_features=df_final_set[['polarity', 'subjectivity', 'first_per_pron', 'sec_per_pron', 'Third_per_pron', 'personal_pron', 'total_pron', 'sing_proper_noun', 'trivial_score', 'negative_word_count', 'positive_word_count', 'vander_score']]
print("Features:",features)
print("Final fe",final_features.head(50))

#Creating Training and Test Data
#df_train_set['is_train'] = np.random.uniform(0, 1, len(df_train_set)) <= .75
#print(df['is_train'])

# Create two new dataframes, one with the training rows, one with the test rows
#train, test = df_train_set[df_train_set['is_train'] == True], df_train_set[df_train_set['is_train'] == False]


#print(train.head())
#print(test.head())


# Show the number of observations for the test and training dataframes
#print('Number of observations in the training data:', len(df_train_set))
#print('Number of observations in the test data:',len(df_test_set))



# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y , y_index  = pd.factorize(df_train_set['sentiment'])
y2 , y2_index= pd.factorize(df_train_set['factual'])
print(y)
print(y_index)
map_sent = list(y_index)
print("map sent",map_sent)
print("printing Y2")
print(y2)
print(y2_index)
map_fact = list(y2_index)
print("map fact",map_fact)
print(len(y))
print(len(y2))


Y=np.column_stack((y,y2))
dict_cls_weight = [{0:1,1:5,2:5},{0:1,1:1}]
#Train The Random Forest Classifier
# Create a random forest classifier. By convention, clf means 'classifier'

sample_wt = sklearn.utils.class_weight.compute_sample_weight(dict_cls_weight,Y)
clf = RandomForestClassifier(n_jobs=2,class_weight=dict_cls_weight)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
#print(df_train_set[features])
#print(df_test_set[features])


clf.fit(df_train_set[features], Y,sample_weight=sample_wt)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
#predict_result =clf.predict(df_test_set[features])
print("before",final_features.describe())

""" tried this part"""
"""groups = final_features.groupby(final_features.index // 1000)
for group in groups:
    predict_result=clf.predict(group.fillna(0))
    print("predict", predict_result)

"""

""" #commenting hardcoded mapping
sentiment_mapping = {
    2: "pos",
    0: "neu",
    1: "neg"
}

factuality_mapping = {
    1: "yes",
    0: "no"
}

"""
sentiment_mapping = {i:map_sent[i] for i in range(len(map_sent))}
factuality_mapping = {i:map_fact[i] for i in range(len(map_fact))}

predict_result=pd.DataFrame()
#predict_result_final=pd.DataFrame()
groups = final_features.groupby(final_features.index // 10000)
for _, group in groups:
    #predict_result = clf.predict(group.fillna(0))
    predict_result = pd.concat([predict_result, pd.DataFrame(clf.predict(group.fillna(0)))], ignore_index=True)



print("predict", predict_result)

print("original",df_final_set_original.head(2))
predict_result.columns = ["sentiment", "factuality"]
print("new sentiment",predict_result['sentiment'])
predict_result["sentiment"] = predict_result["sentiment"].apply(lambda s: sentiment_mapping[s])
predict_result["factuality"] = predict_result["factuality"].apply(lambda f: factuality_mapping[f])
final_visual=pd.concat([predict_result,df_final_set_original[['subforum','post_id','timestamp','author_id','url','thread_id','thread_name','position_in_thread','agrees','sentence','treatments'] ]],axis=1)
print("visual",final_visual)
final_visual.to_csv(data_dir + "/sentences_classified.csv", encoding='utf-8', index=False)





"""
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
for i in range(len(df_final_set)):
    print(y_index[C[i]])
    y_pred.append(y_index[C[i]])
#print(preds2)
y_pred =np.asarray(y_pred)
print(y_pred)
print("Test ",df_test_set['sentiment'])


#predicted values for factual
y2_pred = []
for i in range(len(df_final_set)):
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

"""
