import pandas as pd
import os
import sklearn
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

#-------------training section ---------#

#dataload
raw = pd.read_csv("train.csv")

#dataset inspection
len(raw)
raw.info()
raw.dtypes
summary = raw.describe()

#replacing na values with dummy values
raw.storeId.fillna('0', inplace=True)
raw.url.fillna('Unspecified', inplace=True)
raw.additionalAttributes.fillna('Unspecified', inplace=True)
raw.breadcrumbs.fillna('Unspecified',inplace=True)

raw.info()

#removing characters from storeId
raw['storeId'] = raw['storeId'].map(lambda x: x.lstrip('#'))

#removing long strings from storeid column, planning to treat as integer
raw.loc[raw.storeId.str.len()>5, 'storeId'] = '0'

#removing numericals from url column
raw.loc[raw.url.str.len()<3, 'url'] = 'Unspecified'
raw['url'] = raw.url.str.replace(r'(^.\d.*$)', 'Unspecified')

#removing words which will probably mislead the classifier
raw['url'] = raw.url.str.replace(r'(http://)', '')
raw['url'] = raw.url.str.replace(r'(https://)', '')
raw['url'] = raw.url.str.replace(r'(www.)', '')
raw['url'] = raw.url.str.replace(r'(.com)', '')

#summary visualization
label_summary = raw['label'].value_counts()
#label_summary_plot = label_summary.plot.bar()


#consolidating all features to one variable
raw['features'] = raw['storeId'].map(str) + ' ' +  raw['breadcrumbs']  + ' ' + raw['url'] + ' ' + raw['additionalAttributes']

#assigning dummy labels
raw['label_num'] = raw.label.map({'rest':0, 'books':1, 'music':2, 'videos':3})

#assigning features and labels
x = raw.features
y = raw.label_num

#train-test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

#CountVectorizer
cv = CountVectorizer(stop_words='english')
x_cv = cv.fit_transform(x)

x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

#Classification
mnb = MultinomialNB()
mnb.fit(x_train_cv,y_train)
y_pred = mnb.predict(x_test_cv)

#Model Evaluation
from sklearn.metrics import accuracy_score,confusion_matrix
score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

#K-fold cross validation
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=15,shuffle=True)
cross_val_score_list = [mnb.fit(x_cv[train], y[train]).score(x_cv[test], y[test]) for train, test in k_fold.split(x_cv,y)]
avg_accuracy = sum(cross_val_score_list)/len(cross_val_score_list)


#most informative features - function definition
def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=20):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print classlabel, feat, coef
    return

#displaying the most important features
for item in list(raw.label_num.unique()):
    temp = raw.loc[raw['label_num'] == item, 'label'].iloc[0]
    print "Item : %s " %temp
    most_informative_feature_for_class(cv,mnb,item)


#---------------- predictions ------------------#
eval_set = pd.read_csv('evaluation.csv')

#dataset inspection
len(eval_set)
eval_set.info()
eval_set.dtypes
summary = eval_set.describe()

#replacing na values with dummy values
eval_set.storeId.fillna('0', inplace=True)
eval_set.url.fillna('Unspecified', inplace=True)
eval_set.additionalAttributes.fillna('Unspecified', inplace=True)
eval_set.breadcrumbs.fillna('Unspecified',inplace=True)

eval_set.info()

#removing characters from storeId
eval_set['storeId'] = eval_set['storeId'].map(lambda x: x.lstrip('#'))

#removing long strings from storeid column, planning to treat as integer
eval_set.loc[eval_set.storeId.str.len()>5, 'storeId'] = '0'

#removing numericals from url column
eval_set.loc[eval_set.url.str.len()<3, 'url'] = 'Unspecified'
eval_set['url'] = eval_set.url.str.replace(r'(^.\d.*$)', 'Unspecified')

#removing words which will proabaly mislead the classifier
eval_set['url'] = eval_set.url.str.replace(r'(http://)', '')
eval_set['url'] = eval_set.url.str.replace(r'(https://)', '')
eval_set['url'] = eval_set.url.str.replace(r'(www.)', '')
eval_set['url'] = eval_set.url.str.replace(r'(.com)', '')


eval_set['features'] = eval_set['storeId'].map(str) + ' ' +  eval_set['breadcrumbs']  + ' ' + eval_set['url'] + ' ' + eval_set['additionalAttributes']

#assigning features and labels
x_eval = eval_set.features

#transforming to match dimensions
x_eval_cv = cv.transform(x_eval)

#validating dimensions
x_eval_cv.shape
x_train_cv.shape
x_test_cv.shape

#predictions
y_eval = mnb.predict(x_eval_cv)

df_result = pd.DataFrame({'id':eval_set['id'],'label_num':y_eval})

#remapping numbers to classes
df_result['label'] = df_result.label_num.map({0:'rest', 1:'books', 2:'music', 3:'videos'})

df_result=df_result.drop('label_num', axis=1)

summary_eval = df_result['label'].value_counts()

df_result.to_csv('submissions.csv',index=False)

#eval_set['label'] = df_result['label']

#eval_set.to_csv('eval_set.csv',index=False)
