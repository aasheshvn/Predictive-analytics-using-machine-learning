#!/usr/bin/env python
# coding: utf-8

# In[224]:


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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# ### Importing the dataset

# In[225]:


###Importing the dataset

df=pd.read_csv("energydata_complete.csv")


# In[226]:


### Splitting classes based on its Median

df['Appliances_class'] = [0 if x <= 60 else 1 for x in df['Appliances']] 
# df_logistic = df.drop(columns = ['Appliances'])


# In[227]:


df.columns


# In[228]:


df['Appliances_class'].value_counts()


# In[229]:


pd.DataFrame(df['lights'].value_counts()).T


# ### Feature 'lights' can be removed as it has most of the data points to be 0

# In[231]:


df.drop(columns=['Appliances','lights','date'],inplace=True)


# In[232]:


df.columns


# In[233]:


corr=df.corr()
print("Features with high correlation >0.85\n")
print("Feature1\tFeature2\tCorrelation")

for i in range(len(corr.columns)):
    for j in range(i):
        if(corr.iloc[i,j]>0.85):
            print(corr.columns[i],"\t\t",corr.columns[j],"\t\t", round(corr.iloc[i,j],3))
 


# In[234]:


# T3, RH_4, T5, T8, RH_7, T7, RH_8, T6, rv2           
df.drop(columns=['T3','RH_4','T5','T8','RH_7','T7','RH_8','T6','rv2','Visibility','rv1'],inplace=True)


# In[235]:


df.shape


# ### Getting the count of outliers from each column

# In[236]:


q1= df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3-q1   ###Inter-Quartile Range

upper_threshold = q3 + (1.5 * IQR)
lower_threshold = q1 - (1.5 * IQR)

outliers = dict()
column = 0
for upper,lower in zip(upper_threshold,lower_threshold):
    outliers_count = df[df.iloc[:,column]>math.ceil(upper)].shape[0] + df[df.iloc[:,column]<math.floor(lower)].shape[0]
    outliers[df.columns[column]] = outliers_count
    column = column + 1


# In[237]:


outliers


# In[238]:


df = df[(df["RH_5"] > 33.005) & (df["RH_5"] < 66.058333)]


# ### Feature Scaling

# In[239]:


def feature_scaling(df_unscaled):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_unscaled)
    scaled_data = pd.DataFrame(scaled_data, columns = df_unscaled.columns)
    return(scaled_data)


# ### Train-Test Split

# In[240]:


x=df.loc[:, df.columns != 'Appliances_class']
x=feature_scaling(x)
y=df['Appliances_class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[241]:


X_train.shape


# ## Support Vector Machine

# ### Linear Kernel

# ### Grid Search to get the Optimal Cost Hyperparameter

# In[210]:


from sklearn.model_selection import cross_val_score

cost_values = [0.1,1,10,20]
score_list = []
for cost in cost_values:
    clf = SVC(C=cost,kernel='linear',random_state=10)
    scores = cross_val_score(clf, X_train, y_train, cv=3)
    print(cost, scores.mean())
    score_list.append(scores.mean())
    
cost_accuracy_df = pd.DataFrame({'Accuracy':score_list},index=cost_values)   


# ### Plot for Penalty Vs Accuracy

# In[244]:


plt.plot(cost_accuracy_df,marker='o')
plt.xlabel('Penalty Values',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.title('Penalty Vs Accuracy',fontsize=20)
plt.ylim(0.7,0.75)
plt.show()


# ### We see very slight changes in the accuracies with varying cost. We choose cost 0.1 as the optimal cost hyperparameter for the Linear kernel SVM

# ### Learning curves for Linear kernel SVM

# In[212]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=0.1,kernel='linear'), 
    X_train, y_train,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,random_state=1,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[213]:


plt.plot(train_sizes, train_mean, label = 'Training Accuracy')
plt.plot(train_sizes, valid_mean, label = 'Validation Accuracy')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.title("Train Size vs Accuracy")
plt.legend(loc="lower right")
plt.show()


# ### Test Results Predictions

# In[214]:


accuracy=dict()
svclassifier = SVC(C=0.1,kernel='linear')
svclassifier.fit(X_train, y_train)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train)

print("Accuracy of training data is %.2f" %(accuracy_score(y_train,pred_train)))
print("Accuracy of test data is %.2f" %(accuracy_score(y_test,pred_test)))
accuracy['Linear'] = [accuracy_score(y_train,pred_train),accuracy_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### Polynomial Kernel

# In[215]:


score_list = []
cost_values = [0.1,1,10,20]
degree_values = [1,2,3,4]
for cost in cost_values:
    for degree in degree_values:
        clf = SVC(C=cost,kernel='poly',degree=degree,gamma='auto',random_state=12)
        scores = cross_val_score(clf, X_train, y_train, cv=3)
        print(cost, scores.mean())
        score_list.append(scores.mean())


# In[216]:


cost_values = [0.1,1,10,20]
degree_values = [1,2,3,4]
cost_values = [0.1,1,10,20]*4
cost_values.sort()
degree_values = [1,2,3,4]*4

cost_degree_accuracy = pd.DataFrame({'Degree':degree_values,'Cost':cost_values,'Accuracy':score_list})


# ### Accuracy with varying penalities and degree

# In[219]:


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
ax = sns.lineplot(data=cost_degree_accuracy, x='Cost', y='Accuracy',hue='Degree',palette=flatui)
ax.set(xlabel='Penalty', ylabel='Accuracy')
plt.rcParams["axes.labelsize"] = 20
plt.title('Accuracy Vs (Degree and Penalty)',fontsize=20)
plt.show()


# ### Learning curves for Polynomial kernel SVM

# In[64]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=10,kernel='poly',degree=4,gamma='auto'), 
    X_train, y_train,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,random_state=5,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# ### Test Results Predictions

# In[79]:


svclassifier = SVC(C=20,kernel='poly',degree=4,gamma='auto')
svclassifier.fit(X_train, y_train)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train)

print("Accuracy of training data is %.2f" %(accuracy_score(y_train,pred_train)))
print("Accuracy of test data is %.2f" %(accuracy_score(y_test,pred_test)))
accuracy['Polynomial'] = [accuracy_score(y_train,pred_train),accuracy_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[65]:


plt.plot(train_sizes, train_mean, label = 'Training Accuracy')
plt.plot(train_sizes, valid_mean, label = 'Validation Accuracy')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.title("Train Size vs Accuracy",fontsize=20)
plt.legend(loc="lower right")
plt.show()


# ### Radial Basis Function

# In[220]:


score_list = []
cost_values = [0.1,1,10,20]
gamma_values = [0.01,0.05,0.001,0.005]
for cost in cost_values:
    for gamma in gamma_values:
        clf = SVC(C=cost,kernel='rbf',gamma=gamma)
        scores = cross_val_score(clf, X_train, y_train, cv=3)
        print(cost, scores.mean())
        score_list.append(scores.mean())


# In[67]:


cost_values = [0.1,1,10,20]
gamma_values = [0.01,0.05,0.001,0.005]
cost_values = [0.1,1,10,20]*4
cost_values.sort()
gamma_values = [0.01,0.05,0.001,0.005]*4

cost_gamma_accuracy = pd.DataFrame({'Gamma':gamma_values,'Cost':cost_values,'Accuracy':score_list})


# In[72]:


import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
sns.lineplot(data=cost_gamma_accuracy, x='Cost', y='Accuracy',hue='Gamma',palette=flatui)
plt.xlabel("Penalty",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.legend(loc="lower right")
plt.show()


# ### Learning curves for Radial kernel SVM

# In[73]:


train_sizes, train_scores, valid_scores = learning_curve(SVC(C=20,kernel='rbf',gamma=0.05), 
    X_train, y_train,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,random_state=5,shuffle=True)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)


# In[76]:


plt.plot(train_sizes, train_mean, label = 'Training Accuracy')
plt.plot(train_sizes, valid_mean, label = 'Validation Accuracy')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.legend(loc="lower right")
plt.title("Accuracy vs Train Size",fontsize=20)
plt.show()


# ### Test Results Predictions

# In[77]:


accuracy=dict()
svclassifier = SVC(C=20,gamma=0.05,kernel='rbf')
svclassifier.fit(X_train, y_train)

pred_test = svclassifier.predict(X_test)
pred_train = svclassifier.predict(X_train)

print("Accuracy of training data is %.2f" %(accuracy_score(y_train,pred_train)))
print("Accuracy of test data is %.2f" %(accuracy_score(y_test,pred_test)))
accuracy['Radial'] = [accuracy_score(y_train,pred_train),accuracy_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### Decision Trees

# In[80]:


x=df.loc[:, df.columns != 'Appliances_class']
y=df['Appliances_class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[81]:


X_train.head()


# ### HyperParameter Tuning to choose the optimal Tree depth and Maximum Features

# In[110]:


decision_tree_df=pd.DataFrame()
depth_values = [3,4,5,6,7,8,9,10,11,12,13,14,15]
max_features = [0.2,0.4,0.6,0.8,1]
for depth in depth_values:
    for features in max_features:
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=depth,max_features=features)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        values  =  [depth,features,scores.mean()]
        values = pd.DataFrame(values).T
        decision_tree_df = pd.concat([decision_tree_df,values])
        
decision_tree_df.columns=['Depth','Max Features','Accuracy']     


# ### Accuracy Vs (Depth and Max_Features)

# In[111]:


import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c","#34495e"]
sns.set(rc={'figure.figsize':(5.7,5.27)})
sns.lineplot(data=decision_tree_df, x='Depth', y='Accuracy',hue='Max Features',palette=flatui)
plt.xlabel("Depth",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.title("Depth vs (Accuracy and Max_Features)",fontsize=15)
plt.show()


# In[113]:


accuracy_dt=dict()
for depth in range(1,15):
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=depth,max_features=0.8)  ###Criteria: Entropy
    clf.fit(X_train,y_train)   ### Model fitting
    pred_train = clf.predict(X_train)  ### Training set Prediction
    pred_test = clf.predict(X_test)    ### Testing set Prediction
    
    accuracy_dt[depth]=[accuracy_score(y_train, pred_train).round(3),accuracy_score(y_test, pred_test).round(4)]
    
accuracy_dt_df = pd.DataFrame(accuracy_dt,index=['Train','Test']).T


# In[126]:


plt.figure(figsize=(5,5))
plt.plot(accuracy_dt_df,marker='o')
plt.xlabel('Depth of the Decision Tree',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title("Accuracies on Train and Test based on depth of the tree",fontsize=12)
plt.legend(['Train','Test'],loc="lower right")
plt.show()


# ### Plot to get the Best Features based on their Information Gain

# In[132]:


clf = DecisionTreeClassifier(criterion='entropy',max_depth=10,max_features=0.8)  ###Criteria: Entropy
clf.fit(X_train,y_train)
best_features=dict(zip(X_train.columns, clf.feature_importances_.round(4)))
best_features=pd.DataFrame(best_features,index=['Importance']).T
plt.figure(figsize=(15,5))
plt.plot(best_features,marker='o')
plt.xlabel('Features',fontsize=20)
plt.ylabel('Importance',fontsize=20)
plt.title('Feature Importance Plot',fontsize=20)
plt.show()


# ### Learning curves for Decision Trees

# In[102]:


train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(criterion='entropy',max_depth=10,max_features=0.8), 
    X_train, y_train,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,shuffle=True,random_state=5)

train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label = 'Training error')
plt.plot(train_sizes, valid_mean, label = 'Validation error')
plt.xlabel('Train Size',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.title('Training Size Vs Accuracy',fontsize=20)
plt.legend(['Train','Test'],loc="lower right")
plt.show()


# ### Test Result Predictions

# In[153]:


# accuracy_decision_tree=dict()
classifier = DecisionTreeClassifier(max_depth=10,max_features=0.8)
classifier.fit(X_train, y_train)

pred_test = classifier.predict(X_test)
pred_train = classifier.predict(X_train)
    
print("Accuracy of training data is %.2f" %(accuracy_score(y_train,pred_train)))
print("Accuracy of test data is %.2f" %(accuracy_score(y_test,pred_test)))
# accuracy_decision_tree['DecisionTree'] = [accuracy_score(y_train,pred_train),accuracy_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# ### Boosting using Adaboost Algorithm

# In[164]:


boosting_df=pd.DataFrame()
learners_values = [5,10,25,50,75,100]
learning_rates = [0.01, 0.1, 0.5, 1]
for lrate in learning_rates:
    for learners in learners_values:
        estimator = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=10,max_features=0.8),
                                       n_estimators=learners,learning_rate=lrate)
        scores = cross_val_score(estimator, X_train, y_train, cv=5)
        values  =  [lrate,learners,scores.mean()]
        values = pd.DataFrame(values).T
        boosting_df = pd.concat([boosting_df,values])
        
boosting_df.columns=['lrate','Learners','Accuracy']  
boosting_df = boosting_df.reset_index()


# ### Accuracy Vs (Learning rate and Number of Learners)

# In[165]:


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
sns.lineplot(data=boosting_df, x='Learners', y='Accuracy',hue='lrate',palette=flatui)
plt.xlabel("Number of Learners",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.title("Learners vs Accuracy",fontsize=20)
plt.legend(loc="lower right")
plt.show()


# In[166]:


accuracy_dt=dict()
for depth in range(1,15):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=depth,max_features=0.8),
                             n_estimators=50,learning_rate=1)
    clf.fit(X_train,y_train)   ### Model fitting
    pred_train = clf.predict(X_train)  ### Training set Prediction
    pred_test = clf.predict(X_test)    ### Testing set Prediction
    
    accuracy_dt[depth]=[accuracy_score(y_train, pred_train).round(3),accuracy_score(y_test, pred_test).round(4)]
    
accuracy_dt_df = pd.DataFrame(accuracy_dt,index=['Train','Test']).T


# In[167]:


plt.figure(figsize=(5,5))
plt.plot(accuracy_dt_df,marker='o')
plt.xlabel('Depth of the Decision Tree',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title("Accuracies on Train and Test based on depth of the tree",fontsize=12)
plt.legend(['Train','Test'],loc="lower right")
plt.show()


# ### Learning Curve 

# In[169]:


estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,max_features=0.8),n_estimators=50,
                             learning_rate=1)
train_sizes, train_scores, valid_scores = learning_curve(estimator, X_train, y_train,
                                                         train_sizes=[0.2,0.4,0.6,0.8,1],cv=5,random_state=1,shuffle=True)


# In[170]:


train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label = 'Training Accuracy')
plt.plot(train_sizes, valid_mean, label = 'Validation Accuracy')
plt.xlabel("Train sizes",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.legend(loc="lower right")
plt.title("Accuracy vs Train Size",fontsize=20)
plt.show()


# ### Test Results Predictions

# In[171]:


accuracy_boosting=dict()
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,max_features=0.8),
                                n_estimators=50,learning_rate=1,random_state=1)
classifier.fit(X_train, y_train)

pred_test = classifier.predict(X_test)
pred_train = classifier.predict(X_train)
    
print("Accuracy of training data is %.2f" %(accuracy_score(y_train,pred_train)))
print("Accuracy of test data is %.2f" %(accuracy_score(y_test,pred_test)))
accuracy_boosting['Boosting'] = [accuracy_score(y_train,pred_train),accuracy_score(y_test,pred_test)]
pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[159]:


print(classification_report(y_test,pred_test))


# ### Model comparisons using Cross Validation

# In[175]:


cross_validation_comparisons = pd.DataFrame()
estimators = [SVC(C=20,gamma=0.05,kernel='rbf'),
              DecisionTreeClassifier(criterion='entropy',max_depth=10,max_features=0.8),
              AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=5,max_features=0.8),n_estimators=50,learning_rate=1,random_state=1)]
models = ['Radial SVM', 'Decision Tree', 'Boosted']
for model in range(len(models)):
    scores = cross_val_score(estimators[model], X_train, y_train, cv=5)
    scores = pd.DataFrame(scores).T
    cross_validation_comparisons = pd.concat([cross_validation_comparisons,scores])

cross_validation_comparisons.columns=['Split 1','Split 2','Split 3','Split 4','Split 5'] 
cross_validation_comparisons.index = ['SVC','DecisionTree','BoostedDecisionTree']


# In[178]:


cross_validation_comparisons


# In[192]:


plt.figure(figsize=(8,5))
plt.plot(cross_validation_comparisons.T)
plt.title("Cross Validation with the 3 Algorithms")
plt.xlabel("Split",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
plt.legend(['SVC Radial','DecisionTree','BoostedDecisionTree'])
plt.show()


# In[ ]:




