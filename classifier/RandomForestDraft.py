# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np


# Create a dataframe 
df = pd.read_csv('C:/Users/Atin/Desktop/TestBook1.csv')

#Creating Training and Test Data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
print(df['is_train'])

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


print(train.head())
print(test.head())


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


#Features
#print(df.columns[3:])
features = df.columns[3:]

#print(features)

# train['sentiment'] contains the actual sentiment. So lets encode it.
y , y_index  = pd.factorize(train['sentiment'])
y2 , y2_index= pd.factorize(train['factual'])
# print(y)
# print(y_index)
# print("printing Y2")
# print(y2)
# print(y2_index)
# print(len(y))
# print(len(y2))


Y=np.column_stack((y,y2))

#Train The Random Forest Classifier
clf = RandomForestClassifier(n_jobs=2)

print(train[features])
print(test[features])
clf.fit(train[features], Y)

# prediction result
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
# View the predicted probabilities of  observations
predict_result_prob=clf.predict_proba(test[features])[0:len(test)]
#print(predict_result_prob)
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
