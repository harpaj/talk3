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
from itertools import permutations,combinations


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


#original_col = df_train_set.columns;
#print(original_col)


#1.Pos-Neg Word Count Feature
#For Training Data
df_train_feat1 = sn.defsentfeatextractor(df_train_set)
df_train_set = df_train_set.filter(['post_id', 'sentence','treatments', 'sentiment', 'factual'], axis=1)

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
df_test_set['polarity'] = pd.Series(TextBlob(x).polarity for x in (df_test_set["sentence"].map(str)))


#4. Subjectivity Score
#For Training Data
df_train_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_train_set["sentence"].map(str)))

#For Test Data
df_test_set['subjectivity'] = pd.Series(TextBlob(x).subjectivity for x in (df_test_set["sentence"].map(str)))


#5. Personal Pronoun counts
#Train Data
df_pron_count_train = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_train_set["sentence"].map(str) ),columns=['first_per_pron','sec_per_pron','Third_per_pron','personal_pron','total_pron','sing_proper_noun'])
df_train_set = pd.concat((df_train_set, df_pron_count_train), axis=1)

#For New Data
df_pron_count_test_set = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_test_set["sentence"].map(str) ), columns=['first_per_pron', 'sec_per_pron', 'Third_per_pron', 'personal_pron', 'total_pron', 'sing_proper_noun'])
df_test_set = pd.concat((df_test_set, df_pron_count_test_set), axis=1)

#5. Trivial Score
#Train Data
df_train_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_train_set["sentence"].map(str)))
#For New Data
df_test_set['trivial_score']= pd.Series((sn.get_trivial_score(x) for x in (df_test_set["sentence"].map(str))))

#Merging Feature
#Training Data
df_train_set = pd.concat((df_train_set, df_train_feat1), axis=1)

#Test Data
df_test_set = pd.concat((df_test_set, df_test_feat1), axis=1)

#print("concatenated train data:", df_train_set.head())
#print("concatenated test data:", df_test_set.head())


#Preprocess Data
# Create a list of the feature column's names
global ls_result
ls_result=[]
features_k = df_train_set.columns[5:]
#print(df_train_set.columns)
#print(features_k)
df_csv = pd.DataFrame(columns=['Features','Feature_importance','Sentiment-Accuracy','Factual Accuracy'])
#features_k = pd.DataFrame(columns=['polarity','subjectivity'])
#print(features_k.columns)

#print(features_k)

for L in range(3, len(features_k)+1):
    for subset in combinations(features_k, L):

        #features_k = df_train_set.columns[m:n+1]
        #assert isinstance(subset, object)
        #print(list(subset))
        #print("this",list(features_k))
        temp_df = pd.DataFrame(columns=list(subset))
        features = temp_df.columns
        if len(features)>0:
            #print(features)
            #features = df_train_set.columns[7:]



            # Encoding
            y , y_index  = pd.factorize(df_train_set['sentiment'])
            y2 , y2_index= pd.factorize(df_train_set['factual'])
            #print(y)
            #print(y_index)
            map_sent = list(y_index)
            #print("map sent",map_sent)
            #print("printing Y2")
            #print(y2)
            #print(y2_index)
            map_fact = list(y2_index)
            #print("map fact",map_fact)
            #print(len(y))
            #print(len(y2))


            Y=np.column_stack((y,y2))
            dict_cls_weight = [{0:1,1:5,2:5},{0:1,1:1}]

            #Train The Random Forest Classifier
            # Create a random forest classifier. By convention, clf means 'classifier'

            sample_wt = sklearn.utils.class_weight.compute_sample_weight(dict_cls_weight,Y)
            clf = RandomForestClassifier(n_jobs=1,class_weight=dict_cls_weight,random_state=67)

            # Train the classifier to take the training features and learn how they relate
            # to the training y (the species)
            #print(df_train_set[features])
            #print(df_test_set[features])


            clf.fit(df_train_set[features], Y,sample_weight=sample_wt)

            predict_result = clf.predict(df_test_set[features])

            #print("Feature Importance:",list(clf.feature_importances_))

            df_feature_imp = pd.DataFrame(columns=(features))
            df_feature_imp = df_feature_imp.append(pd.Series(list(clf.feature_importances_), index=df_feature_imp.columns), ignore_index=True)
            #print("Feat_imp:")
            #print(df_feature_imp)
            #Evaluate classifier
            #encoded value for sentiment
            A = np.array(predict_result)
            B=np.asmatrix(A)
            C=B[:,0]
            C=C.astype(int)
            #print(C)


            C=C.ravel()
            C = np.asarray(C)
            C=C[0]
            #print(C)

            #encoded value for factuality
            B=np.asmatrix(A)
            C2=B[:,1]
            C2=C2.astype(int)
            #print(C2)


            C2=C2.ravel()
            C2 = np.asarray(C2)
            C2=C2[0]
            #print(C2)

            #------------------------------------------------------------------
            # View the predicted probabilities of the first 10 observations
            #predict_result_prob=clf.predict_proba(df_test_set[features])[0:len(df_test_set)]
            #print(predict_result_prob)
            #--------------------------------------------------------------------

            #Evaluate classifier
            #predicted values for sentiment
            y_pred = []
            for i in range(len(df_test_set)):
                #print(y_index[C[i]])
                y_pred.append(y_index[C[i]])
            #print(preds2)
            y_pred =np.asarray(y_pred)
            #print(y_pred)
            #print("Test ",df_test_set['sentiment'])


            #predicted values for factual
            y2_pred = []
            for i in range(len(df_test_set)):
                #print(y2_index[C2[i]])
                y2_pred.append(y2_index[C2[i]])
            #print(preds2)
            y2_pred =np.asarray(y2_pred)
            #print(y2_pred)
            #print("Test ",df_test_set['factual'])


            # Create confusion matrix
            confusion_matrix=pd.crosstab(df_test_set['sentiment'], y_pred, rownames=['Actual Sentiment'], colnames=['Predicted Sentiment'],margins=True)

            if 'neu' not in confusion_matrix.columns:
                #print("problem")
                neu = 0
            else:
                neu = confusion_matrix["neu"]["neu"]

            if 'neg' not in confusion_matrix.columns:
                #print("problem")
                neg = 0
            else:
                neg = confusion_matrix["neg"]["neg"]

            if 'pos' not in confusion_matrix.columns:
                #print("problem")
                pos = 0
            else:
                pos = confusion_matrix["pos"]["pos"]



            sentiment_accuracy = (neg+neu+pos)/len(df_test_set)




            # Create confusion matrix
            confusion_matrix=pd.crosstab(df_test_set['factual'], y2_pred, rownames=['Actual Factuality'], colnames=['Predicted Factuality'],margins=True)

            if 'no' not in confusion_matrix.columns:
                #print("problem")
                no = 0
            else:
                no = confusion_matrix["no"]["no"]

            if 'yes' not in confusion_matrix.columns:
                #print("problem")
                yes = 0
            else:
                yes = confusion_matrix["yes"]["yes"]

            factual_accuracy = (no+yes)/len(df_test_set)

            del confusion_matrix
            res = [list(features),list(clf.feature_importances_),sentiment_accuracy,factual_accuracy]
            print(res)
            df_csv = df_csv.append(pd.Series(res,index=df_csv.columns),ignore_index=True)
            #ls_result.__add__(res)


#ls_result.sort(key=lambda tup: tup[1])
df_csv.to_csv(data_dir +'/feature_test.csv',index=False)
#print(ls_result)

