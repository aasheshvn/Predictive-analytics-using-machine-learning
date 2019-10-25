#!/usr/bin/env python
# coding: utf-8

# In[172]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# ### Importing the Dataset

# In[177]:


df = pd.read_csv("UCI_Credit_Card.csv")
df=df.drop(columns='ID')
df = df.rename(columns = {'default.payment.next.month':'default_class'})
df.head()


# ### Data Exploration and Data Cleaning

# In[183]:


print(pd.DataFrame(df['default_class'].value_counts()))
print("\n",pd.DataFrame(df['SEX'].value_counts()))
print("\n",pd.DataFrame(df['EDUCATION'].value_counts()))
print("\n",pd.DataFrame(df['MARRIAGE'].value_counts()))


# ### Default Class type = 1 is only 22% of the entire dataset. It is a highly unbalanced dataset.

# #### Education categories 4, 5, 6, 0 can be grouped under 'Other' category

# In[133]:


df.loc[(df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0),'EDUCATION'] = 4
pd.DataFrame(df['EDUCATION'].value_counts())


# #### Marriage categories 3, 0 can be grouped under 'Other' category

# In[135]:


df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
pd.DataFrame(df['MARRIAGE'].value_counts())


# ### Dummy Variables creation for categorical variables

# In[185]:


df = pd.get_dummies(df, columns=['SEX', 'EDUCATION','MARRIAGE'])


# ### Train Test Split

# In[187]:


x=df.loc[:,df.columns != 'default_class']
y=df['default_class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# ### Upsampling the Default Class 1

# In[190]:


xy_train = pd.concat([X_train, y_train], axis=1)
not_default = xy_train[xy_train['default_class']==0]
default = xy_train[xy_train['default_class']==1]

# upsample minority
default_upsampled_df = resample(default, replace=True, n_samples=len(not_default), random_state=7) 
upsampled_df = pd.concat([not_default, default_upsampled_df])
pd.DataFrame(upsampled_df['default_class'].value_counts())


# In[191]:


X_train_upsampled = upsampled_df.loc[:,upsampled_df.columns != 'default_class']
y_train_upsampled = upsampled_df['default_class']


# ### Under sampling the Default Class 0

# In[192]:


xy_train = pd.concat([X_train, y_train], axis=1)
not_default = xy_train[xy_train['default_class']==0]
default = xy_train[xy_train['default_class']==1]

default_undersampled_df = resample(not_default, replace=True, n_samples=7417, random_state=7) 
undersampled_df = pd.concat([default, default_undersampled_df])
pd.DataFrame(undersampled_df['default_class'].value_counts())


# In[193]:


X_train_undersampled = undersampled_df.loc[:,undersampled_df.columns != 'default_class']
y_train_undersampled = undersampled_df['default_class']
X_train_undersampled.head()


# ### Decision Trees

# ### Hyperparameter tuning to find optimal value for max_features and min_samples_split

# In[194]:


decision_tree_df=pd.DataFrame()
min_samples_split = [2,4,6,8,10,12,14]
max_features = [0.2,0.4,0.6,0.8,1]
for min_split in min_samples_split:
    for features in max_features:
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=15,min_samples_split=min_split,max_features=features,
                                    class_weight = 'balanced',random_state = 3)
        scores = cross_val_score(clf, X_train_upsampled, y_train_upsampled, cv=5,scoring= 'f1')
        values  =  [min_split,features,scores.mean()]
        values = pd.DataFrame(values).T
        decision_tree_df = pd.concat([decision_tree_df,values])
        
decision_tree_df.columns=['Minimum Samples Split','Max Features','F1 Score']     


# In[202]:


import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c","#34495e"]
sns.set(rc={'figure.figsize':(5.7,5.27)})
sns.lineplot(data=decision_tree_df, x='Minimum Samples Split', y='F1 Score',hue='Max Features',palette=flatui)
plt.xlabel("Minimum Samples Split",fontsize=20)
plt.ylabel("F1 Score",fontsize=20)
plt.title("F1 Score vs (Minimum Samples Split and Max_Features)",fontsize=13)
plt.show()


# #### Optimal Minimum_samples_Split = 2 or 4 , Optimal Max_Features = 0.8

# ### Learning Curves with Training sizes

# In[272]:


train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(criterion='entropy',
                            max_depth=10,max_features=0.8,min_samples_split=2,class_weight = 'balanced',random_state = 3), 
    X_train_upsampled, y_train_upsampled,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,shuffle=True,random_state=5,scoring='f1')

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[273]:


plt.plot(train_sizes, train_mean, label = 'Training F1 Score')
plt.plot(train_sizes, valid_mean, label = 'Validation F1 Score')
plt.xlabel("Train Size",fontsize=20)
plt.ylabel("F1 Score",fontsize=20)
plt.legend(['Train','Test'])
plt.title("F1 Score vs Train Size",fontsize=13)
plt.show()
plt.show()


# ### Experimentation with pruning

# In[274]:


f1_score_dt=dict()
for depth in range(1,15):
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=depth,max_features=0.8,min_samples_split=4,
                                 class_weight = 'balanced',random_state=3)  
    clf.fit(X_train_upsampled,y_train_upsampled)   ### Model fitting
    pred_train = clf.predict(X_train_upsampled)  ### Training set Prediction
    pred_test = clf.predict(X_test)    ### Testing set Prediction
    
    f1_score_dt[depth]=[f1_score(y_train_upsampled, pred_train).round(3),f1_score(y_test, pred_test).round(4)]
    
f1_score_dt_df = pd.DataFrame(f1_score_dt,index=['Train','Test']).T


# In[275]:


plt.figure(figsize=(5,5))
plt.plot(f1_score_dt_df)
plt.xlabel('Depth of the Decision Tree',fontsize=20)
plt.ylabel('F1 score',fontsize=20)
plt.legend(['Train','Test'])
plt.title('F1 Score Vs Tree Depth',fontsize=20)
plt.show()


# In[277]:


clf = DecisionTreeClassifier(criterion='entropy',max_depth=4,max_features=0.8,
                             class_weight = 'balanced',min_samples_split=4,random_state=3)  ###Criteria: Entropy
clf.fit(X_train_upsampled,y_train_upsampled)
best_features=dict(zip(X_train_upsampled.columns, clf.feature_importances_.round(4)))
best_features=pd.DataFrame(best_features,index=['Importance']).T
plt.figure(figsize=(5,4))
plt.plot(best_features.sort_values(by = 'Importance',ascending=False)[:5],marker='o')
plt.xlabel('Features',fontsize=20)
plt.ylabel('Information',fontsize=20)
plt.title("Feature Importance",fontsize=20)
plt.show()


# ### Test Dataset Predictions

# In[278]:


clf = DecisionTreeClassifier(criterion='entropy',max_depth=4,max_features=0.8,
                             min_samples_split=2,class_weight = 'balanced',random_state=3)  
clf.fit(X_train_upsampled,y_train_upsampled)   ### Model fitting
pred_train = clf.predict(X_train_upsampled)  ### Training set Prediction
pred_test = clf.predict(X_test)    ### Testing set Prediction
print(classification_report(y_test,pred_test))
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# #### The performance on the test F1 score is not improving with increasing depth. But the performances looks good on training set. This is because the training dataset is balanced and the test set is highly unbalanced

# ### Boosting using Adaboost Algorithm

# In[227]:


boosting_df=pd.DataFrame()
learners_values = [5,50,100,200]
for learners in learners_values:
        estimator = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=4,max_features=0.8,
                             min_samples_split=2,class_weight = 'balanced',random_state=3),n_estimators=learners)
        scores = cross_val_score(estimator, X_train_upsampled, y_train_upsampled, cv=5,scoring='f1')
        values  =  [learners,scores.mean()]
        values = pd.DataFrame(values).T
        boosting_df = pd.concat([boosting_df,values])
        
boosting_df.columns=['Learners','F1_Score']  
boosting_df = boosting_df.reset_index()


# In[229]:


sns.lineplot(data=boosting_df, x='Learners', y='F1_Score')
plt.xlabel('Number of estimators',fontsize=20)
plt.ylabel('F1 Score',fontsize=20)
plt.title("F1 Score Vs Number of estimators",fontsize=20)
plt.show()


# In[280]:


f1_score_dt=dict()
for depth in range(1,15):
    estimator = DecisionTreeClassifier(criterion='entropy',max_depth=depth,max_features=0.8,min_samples_split=4,
                           class_weight = 'balanced')  
    clf = AdaBoostClassifier(estimator,n_estimators=50)  
    clf.fit(X_train_undersampled,y_train_undersampled)   ### Model fitting
    pred_train = clf.predict(X_train_undersampled)  ### Training set Prediction
    pred_test = clf.predict(X_test)    ### Testing set Prediction
    
    f1_score_dt[depth]=[f1_score(y_train_undersampled, pred_train).round(3),f1_score(y_test, pred_test).round(4)]
    
f1_score_dt_df = pd.DataFrame(f1_score_dt,index=['Train','Test']).T


# In[282]:


plt.figure(figsize=(5,5))
plt.plot(f1_score_dt_df)
plt.xlabel('Depth of the Decision Tree',fontsize=20)
plt.ylabel('F1 score',fontsize=20)
plt.legend(['Train','Test'])
plt.show()


# ### Learning Curve

# In[292]:


estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,max_features=0.8,random_state=3),
                                    n_estimators=50,random_state=8)
train_sizes, train_scores, valid_scores = learning_curve(estimator,
                            X_train_upsampled, y_train_upsampled,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[295]:


plt.plot(train_sizes, train_mean, label = 'Training error')
plt.plot(train_sizes, valid_mean, label = 'Validation error')
plt.xlabel('Train Size',fontsize=20)
plt.ylabel('F1 score',fontsize=20)
plt.legend(['Train','Validation'])
plt.title("F1 score Vs Training Size", fontsize=15)
plt.show()


# ### Test Dataset Predictions

# In[251]:


classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,
                                                       max_features=0.8,min_samples_split=4,random_state=3),
                                n_estimators=50,random_state=9)
classifier.fit(X_train_upsampled, y_train_upsampled)

pred_test = classifier.predict(X_test)
pred_train = classifier.predict(X_train_upsampled)
    
print(classification_report(y_test,pred_test))
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### Feature Scaling

# In[252]:


def feature_scaling(df_unscaled):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_unscaled)
    scaled_data = pd.DataFrame(scaled_data, columns = df_unscaled.columns)
    return(scaled_data)


# ### Preparing the dataset for SVM

# In[253]:


x=df.loc[:,df.columns != 'default_class']
x.iloc[:,0:20]=feature_scaling(x.iloc[:,0:20])
y=df['default_class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# ### Undersampling the dataset to 60 to 40 ratio

# In[263]:


xy_train = pd.concat([X_train, y_train], axis=1)
not_default = xy_train[xy_train['default_class']==0]
default = xy_train[xy_train['default_class']==1]

# upsample minority
default_undersampled_df = resample(not_default, replace=True, n_samples=7417, random_state=17) 
undersampled_df = pd.concat([default, default_undersampled_df])
undersampled_df['default_class'].value_counts()

X_train_undersampled = undersampled_df.loc[:,undersampled_df.columns != 'default_class']
y_train_undersampled = undersampled_df['default_class']


# In[264]:


X_train_undersampled.head()


# ### SVM Linear Kernel

# In[256]:


from sklearn.model_selection import cross_val_score

cost_values = [0.01,0.1,0.5,1]
score_list = []
for cost in cost_values:
    clf = SVC(C=cost,kernel='linear',random_state=15)
    scores = cross_val_score(clf, X_train_undersampled, y_train_undersampled, cv=3,scoring='f1')
    print(cost, scores.mean())
    score_list.append(scores.mean())
    
cost_accuracy_df = pd.DataFrame({'F1 Score':score_list},index=cost_values)   


# ### Plot for Penalty Vs F1 Score

# In[298]:


plt.plot(cost_accuracy_df,marker='o')
plt.xlabel('Penalty Values',fontsize=20)
plt.ylabel('F1 score',fontsize=20)
plt.ylim(0.5,0.7)
plt.show()


# ### Learning curves for Linear kernel SVM

# In[265]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=0.1,kernel='linear'), 
    X_train_undersampled, y_train_undersampled,train_sizes=[0.2,0.4,0.6,0.8,1],scoring='f1',cv=5,
                                                         random_state=2,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label = 'Training F1 score')
plt.plot(train_sizes, valid_mean, label = 'Validation F1 score')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("F1 Scores",fontsize=20)
plt.title("Train Size vs F1 Score")
plt.legend(loc="lower right")
plt.show()


# ### Test Dataset Predictions

# In[266]:


f1_score_dic=dict()
svclassifier = SVC(C=0.1,kernel='linear')
svclassifier.fit(X_train_undersampled, y_train_undersampled)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train_undersampled)

print("F1 score of training data is %.2f" %(f1_score(y_train_undersampled,pred_train)))
print("F1 Score of test data is %.2f" %(f1_score(y_test,pred_test)))
f1_score_dic['Linear'] = [f1_score(y_train_undersampled,pred_train),f1_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### SVM - Polynomial Kernel

# In[267]:


svm_poly_df = pd.DataFrame()
cost_values = [0.01,0.1,0.5,1]
degree_values = [1,2,3,4]
for cost in cost_values:
    for degree in degree_values:
        clf = SVC(C=cost,kernel='poly',degree=degree,gamma='auto')
        scores = cross_val_score(clf, X_train_undersampled, y_train_undersampled, cv=3,scoring='f1')
        print(cost, scores.mean())
        values  =  [cost,degree,scores.mean()]
        values = pd.DataFrame(values).T
        svm_poly_df = pd.concat([svm_poly_df,values])
        
svm_poly_df.columns=['Cost','Degree','F1_Score']     


# ### F1 Score Vs (Degree and Penalty)

# In[300]:


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
ax = sns.lineplot(data=svm_poly_df, x='Cost', y='F1_Score',hue='Degree',palette=flatui)
ax.set(xlabel='Penalty', ylabel='F1 Score')
plt.rcParams["axes.labelsize"] = 20
plt.show()


# ### Learning curve - F1 Score Vs Train Size

# In[270]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=0.5,kernel='poly',degree=2,gamma='auto'), 
    X_train_undersampled, y_train_undersampled,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,scoring='f1',random_state=5,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[271]:


plt.plot(train_sizes, train_mean, label = 'Training F1 Score')
plt.plot(train_sizes, valid_mean, label = 'Validation F1 Score')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("F1 Score",fontsize=20)
plt.title("Train Size vs F1 Score",fontsize=20)
plt.legend(loc="lower right")
plt.show()


# ### Test Dataset Predictions

# In[35]:


f1_score_dic = dict()
svclassifier = SVC(C=0.5,kernel='poly',degree=2,gamma='auto')
svclassifier.fit(X_train_undersampled, y_train_undersampled)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train_undersampled)

print("F1 Score of training data is %.2f" %(f1_score(y_train_undersampled,pred_train)))
print("F1 Score of test data is %.2f" %(f1_score(y_test,pred_test)))
f1_score_dic['Polynomial'] = [f1_score(y_train_undersampled,pred_train),f1_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### Radial Basis Function

# In[285]:


svm_radial_df = pd.DataFrame()
cost_values = [0.01,0.1,0.5,1]
gamma_values = [0.01,0.05,0.001,0.005]
for cost in cost_values:
    for gamma in gamma_values:
        clf = SVC(C=cost,kernel='rbf',gamma=gamma)
        scores = cross_val_score(clf, X_train_undersampled, y_train_undersampled, cv=3)
        print(cost, scores.mean())
        values  =  [cost,gamma,scores.mean()]
        values = pd.DataFrame(values).T
        svm_radial_df = pd.concat([svm_radial_df,values])
        
svm_radial_df.columns=['Cost','Gamma','F1_Score']     


# ### F1 Score Vs (Penalty and Gamma)

# In[286]:


import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
sns.lineplot(data=svm_radial_df, x='Cost', y='F1_Score',hue='Gamma',palette=flatui)
plt.xlabel("Penalty",fontsize=20)
plt.ylabel("F1 Score",fontsize=20)
plt.title("F1 Score Vs (Penalty and Gamma)",fontsize=20)
plt.legend(loc="lower right")
plt.show()


# ### Learning Curve - Train Size Vs F1 Score

# In[287]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=0.5,kernel='rbf',gamma=0.05), 
    X_train_undersampled, y_train_undersampled,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,random_state=5,shuffle=True,scoring='f1')

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[290]:


plt.plot(train_sizes, train_mean, label = 'Training F1 Score')
plt.plot(train_sizes, valid_mean, label = 'Validation F1 Score')
plt.xlabel("Train Size ",fontsize=20)
plt.ylabel("F1 Score",fontsize=20)
plt.title("Training size Vs F1 Score", fontsize=20)
plt.show()


# ### Test Results Predictions

# In[296]:


f1_score_dic = dict()
svclassifier = SVC(C=0.5,kernel='rbf',gamma=0.05)
svclassifier.fit(X_train_undersampled, y_train_undersampled)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train_undersampled)

print("F1 Score of training data is %.2f" %(f1_score(y_train_undersampled,pred_train)))
print("F1 Score of test data is %.2f" %(f1_score(y_test,pred_test)))
f1_score_dic['Radial'] = [f1_score(y_train_undersampled,pred_train),f1_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[50]:


print(classification_report(y_test,pred_test))


# In[ ]:




