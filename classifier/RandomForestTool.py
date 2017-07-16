# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np
from textblob import TextBlob
from featureAnalyzer import feat_analyser as sn


from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

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

#Test Data
df_pron_count_test = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_test_set["sentence"].map(str)),columns=['first_per_pron','sec_per_pron','Third_per_pron','personal_pron','total_pron','sing_proper_noun'])
df_test_set = pd.concat((df_test_set, df_pron_count_test), axis=1)

#Added for Test Set 16-07-2017 Atin
#For New Data
df_pron_count_final_set = pd.DataFrame((list(sn.count_pronouns(x)) for x in df_final_set['sentence'].map(str)),columns=['first_per_pron','sec_per_pron','Third_per_pron','personal_pron','total_pron','sing_proper_noun'])
df_final_set = pd.concat((df_final_set, df_pron_count_final_set), axis=1)

#5. Trivial Score
#Train Data
df_train_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_train_set["sentence"].map(str)))

#Added for Test Set 16-07-2017 Atin
#Test Data
df_test_set['trivial_score'] = pd.Series(sn.get_trivial_score(x) for x in (df_test_set["sentence"].map(str)))

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

#Putting Final 8 Features 16-07-2017 Atin

features = df_train_set.columns[5:]
list_feat = list(features)
list_feat.remove('negative_word_count')
list_feat.remove('Third_per_pron')
list_feat.remove('polarity')
list_feat.remove('subjectivity')
temp_df = pd.DataFrame(columns=list_feat)
features = temp_df.columns


final_features=pd.DataFrame
#changed on 07-07-2017 by Atin
#final_features=df_final_set[['negative_word_count','positive_word_count','polarity','subjectivity','vander_score']]

#Changing to Final 8 feature 16-07-2017 Atin
#final_features=df_final_set[['polarity', 'subjectivity', 'first_per_pron', 'sec_per_pron', 'Third_per_pron', 'personal_pron', 'total_pron', 'sing_proper_noun', 'trivial_score', 'negative_word_count', 'positive_word_count', 'vander_score']]
final_features=df_final_set[[ 'first_per_pron', 'sec_per_pron',  'personal_pron', 'total_pron', 'sing_proper_noun', 'trivial_score',  'positive_word_count', 'vander_score']]
print("Features:",features)
print("Final fe",final_features.head(50))

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

#Changed on 16-07-2017 Atin
#dict_cls_weight = [{0:1,1:5,2:5},{0:1,1:1}]
dict_cls_weight ='balanced'
# Create a random forest classifier. By convention, clf means 'classifier'

sample_wt = sklearn.utils.class_weight.compute_sample_weight(dict_cls_weight,Y)

#Added Random State 16-07-2017 Atin
clf = RandomForestClassifier(n_jobs=1,class_weight=dict_cls_weight,random_state=56)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
#print(df_train_set[features])
#print(df_test_set[features])


clf.fit(df_train_set[features], Y,sample_weight=sample_wt)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
#predict_result =clf.predict(df_test_set[features])
print("before",final_features.describe())

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





