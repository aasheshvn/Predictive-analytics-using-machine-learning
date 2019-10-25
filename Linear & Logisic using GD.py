#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import accuracy_score
import random


# In[3]:


###Importing the dataset

df=pd.read_csv("energydata_complete.csv")


# ### Descriptive Statistics

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


pd.DataFrame(df['lights'].value_counts()).T


# In[8]:


df.corr()


# In[181]:


15252/19735


# ### Checking for columns with high correlation

# In[9]:


corr=df.corr()
print("Features with high correlation >0.88\n")
print("Feature1\tFeature2\tCorrelation")

for i in range(len(corr.columns)):
    #print("i= ", i)
    for j in range(i):
        #print("j= ", j)
        if(corr.iloc[i,j]>0.88):
            print(corr.columns[i],"\t\t",corr.columns[j],"\t\t", round(corr.iloc[i,j],3))
            


# ### Dropping the columns

# In[10]:


df_filter=df.drop(columns =['date','lights','T3','T_out','rv1','rv2','RH_4','T4','T7','RH_7','T5'])


# In[11]:


df_filter.shape


# ### Outliers detection and removal

# In[12]:


### For Appliances
plt.boxplot(df_filter['Appliances'])
plt.title("\nOutliers in Energy Usage\n")
plt.xlabel("Appliances")
plt.ylabel("Energy Usage")
plt.show()


# In[313]:


df['Appliances'].describe()


# In[314]:


plt.hist(df_filter['Appliances'])
plt.title("\nHistogram for Energy Usage\n")
plt.xlabel("Energy Usage")
plt.ylabel("Count")
plt.show()


# In[315]:


df_filter[df_filter['Appliances']>175].shape


# In[316]:


2138/19735


# In[13]:


def outlier(df_with_outliers):
    df_no_outliers = df_with_outliers[df_with_outliers['Appliances']<175]
    return(df_no_outliers)


# ### Feature Scaling

# In[14]:


def feature_scaling(df_unscaled):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_unscaled)
    scaled_data = pd.DataFrame(scaled_data, columns = df_unscaled.columns)
    return(scaled_data)


# ### Accuracy Metrics Calculation

# In[15]:


def calculate_metrics(actual,predicted,rows):
    cf=round((((predicted-actual)**2).sum())/(2*rows),4)   ### Cost
    mse = round((np.mean((actual - predicted)**2)),4)  ### Mean Squared Error
    mae = round((np.mean(abs(actual - predicted))),4)       ### Mean Absolute Error
    
    actual_mean = actual.mean()
    tss = np.sum((actual - actual_mean)**2)
    rss = np.sum((actual - predicted)**2)
    
    r_squared = round((1 - (rss/tss)),4)         ### R Squared
    
    return (cf,mse,mae,r_squared)


# ### Gradient Descent for Linear Regression

# In[16]:


###Initialisations

def GD_Linear(X_train, X_test, y_train, y_test,exp):
    if exp == 1:
        lrate=[0.005,0.007,0.01]  ###Learning Rates with increasing step sizes
        threshold=[0.1]
        iterations=1000
    if exp == 2:
        lrate=[0.01]  ###Learning Rates with increasing step sizes
        threshold=[0.005,0.01,0.1]
        iterations=2000
    if exp == 3 or exp == 4:
        lrate=[0.01]
        threshold=[0.1]
        iterations=1000
    beta_dict=dict()
    cf_dict=dict()
    mse_train=dict()
    mae_train=dict()
    r_squared=dict()
    ####

    rows=X_train.shape[0]
    n=X_train.shape[1]+1
    xt=X_train.values.transpose()

    for alpha in lrate:
        for t in threshold:
            p_diff_cost=[0]*n
            beta=[0.5]*n
            cf=[0]*iterations
            mse=[0]*iterations
            mae=[0]*iterations
            r2=[0]*iterations
            flag=0
            for iter in range(iterations):
                for k in range(n):
                    if(k == 0):
                        p_diff_cost[0]=((beta[0]+np.dot(beta[1:],xt))-y_train).sum()
                    else:
                        p_diff_cost[k]=np.dot((beta[0]+(np.dot(beta[1:],xt))-y_train),X_train.iloc[:,k-1])

                for bit in range(n):
                    beta[bit]=round((beta[bit]-(alpha*p_diff_cost[bit])/rows),4)

                pred=beta[0]+np.dot(beta[1:],xt)

                cf[iter],mse[iter],mae[iter],r2[iter] = calculate_metrics(y_train,pred,rows)
                if cf[iter-1] - cf[iter]<=t and flag == 0 and iter>0 and exp == 1:
                    print("With learning rate of %.3f, cost fuction converged at iteration %d" %(alpha,iter))
                    flag=1

                if cf[iter-1] - cf[iter]<=t and exp == 2 and iter>0:
                    print("Convergence with threshold %f reached at iteration %d" %(t,iter))
                    beta_dict[t], cf_dict[t],mse_train[t],mae_train[t],r_squared[t] = beta,cf,mse,mae,r2
                    break

        if exp == 1 or exp==3 or exp == 4:
            beta_dict[alpha], cf_dict[alpha],mse_train[alpha],mae_train[alpha],r_squared[alpha] = beta,cf,mse,mae,r2
        
    return(beta_dict,cf_dict,mse_train,mae_train,r_squared)


# In[17]:


df_no_outliers = outlier(df_filter) ###Outlier Removal
df_scaled = feature_scaling(df_no_outliers)  ###Normalisation of features

x=df_scaled.iloc[:,1:]
y=df_no_outliers['Appliances']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

beta_dict,cf_dict,mse_train,mae_train,r_squared = GD_Linear(X_train, X_test, y_train, y_test,1)


# ### Plots

# In[18]:


for i in [0.005,0.007,0.01]:
    plt.plot(cf_dict[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nCost Function")
plt.title("\nCost Function with different Learning rates\n")
plt.show()


# In[19]:


for i in [0.005,0.007,0.01]:
    plt.plot(mse_train[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nMean Squared Error")
plt.title("\nMean Squared Error with different Learning rates\n")
plt.show()


# In[20]:


for i in [0.005,0.007,0.01]:
    plt.plot(mae_train[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nMean Absolute Error")
plt.title("\nMean Absolute Error with different Learning rates\n")
plt.show()
plt.show()


# ### Best Values for Experimentation 1

# In[21]:


cf_final = {}
mse_final = {}
mae_final = {}
r2_final = {}
for alpha in [0.005,0.007,0.01]:
    cf_final[alpha] = min(cf_dict[alpha])
    mse_final[alpha] = min(mse_train[alpha])
    mae_final[alpha] = min(mae_train[alpha])
    r2_final[alpha] = max(r_squared[alpha])


# In[22]:


train_metrics= pd.DataFrame([cf_final, mse_final,mae_final,r2_final],index=['CF_Train','MSE_Train','MAE_Train','R2_Train']).T

xt_test = X_test.values.transpose()
rows_test = X_test.shape[0]
cf_test = {}
mse_test = {}
mae_test = {}
r2_test = {}
for alpha in [0.005,0.007,0.01]:
    beta = beta_dict[alpha]
    pred_test = beta[0]+np.dot(beta[1:],xt_test)
    cf_test[alpha],mse_test[alpha],mae_test[alpha],r2_test[alpha] = calculate_metrics(y_test,pred_test,rows_test)
    
test_metrics=pd.DataFrame([cf_test, mse_test,mae_test,r2_test],index=['CF_Test','MSE_Test','MAE_Test','R2_Test']).T

metrics1=pd.concat([train_metrics,test_metrics],axis=1)
metrics1.reindex(sorted(metrics1.columns), axis=1)


# In[23]:


metrics1[['CF_Test','CF_Train']].plot(kind='bar', title ="Cost Function", legend=True, fontsize=12)


# ### Experimentation 2 - Thresholds

# In[24]:


beta_dict,cf_dict,mse_train,mae_train,r_squared = GD_Linear(X_train, X_test, y_train, y_test,2)


# In[25]:


for i in [0.005,0.01,0.1]:
    plt.plot(cf_dict[i],label='Threshold: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nCost Function")
plt.title("\nCost Function with different Thresholds\n")
plt.show()


# In[26]:


for i in [0.005,0.01,0.1]:
    plt.plot(mse_train[i],label='Threshold: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nMean Squared Error")
plt.title("\nMean Squared Error with different Thresholds\n")
plt.show()


# ### Best Values for Experimentation 2

# In[27]:


cf_final = {}
mse_final = {}
mae_final = {}
r2_final = {}
for t in [0.005,0.01,0.1]:
    
    cf_dict[t].sort()
    mse_train[t].sort()
    mae_train[t].sort()
    r_squared[t].sort()
    cf_final[t] = min([x for x in cf_dict[t] if x!=0])
    mse_final[t] = min([x for x in mse_train[t] if x!=0])
    mae_final[t] = min([x for x in mae_train[t] if x!=0])
    r2_final[t] = max([x for x in r_squared[t] if x!=0])


# In[28]:


train_metrics= pd.DataFrame([cf_final, mse_final,mae_final,r2_final],index=['CF_Train','MSE_Train','MAE_Train','R2_Train']).T


# In[29]:


xt_test = X_test.values.transpose()
rows_test = X_test.shape[0]
cf_test = {}
mse_test = {}
mae_test = {}
r2_test = {}
for t in [0.005,0.01,0.1]:
    beta = beta_dict[t]
    pred_test = beta[0]+np.dot(beta[1:],xt_test)
    cf_test[t],mse_test[t],mae_test[t],r2_test[t] = calculate_metrics(y_test,pred_test,rows_test)


# In[30]:


test_metrics=pd.DataFrame([cf_test, mse_test,mae_test,r2_test],index=['CF_Test','MSE_Test','MAE_Test','R2_Test']).T


# In[31]:


metrics1=pd.concat([train_metrics,test_metrics],axis=1)
metrics1.reindex(sorted(metrics1.columns), axis=1)


# In[32]:


metrics1[['CF_Test','CF_Train']].plot(kind='bar', title ="Cost Function", legend=True, fontsize=12)


# ### Experimentation 3: Training and Testing on  10 Random Features

# In[34]:


import random
random.seed(1)
random_features=random.sample(range(27), 10)

df=pd.read_csv("energydata_complete.csv")
df=df.drop(columns = ['date'])
df_no_outliers = outlier(df) ###Outlier Removal
df_scaled = feature_scaling(df_no_outliers)  ###Normalisation of features

x_random=df_scaled.iloc[:,random_features]
y=df_no_outliers['Appliances']
X_train3, X_test3, y_train, y_test = train_test_split(x_random, y, test_size=0.3,random_state=1)


# x_random = x.iloc[:,random_features]
# y=df_no_outliers['Appliances']
# X_train3, X_test3, y_train, y_test = train_test_split(x_random, y, test_size=0.3,random_state=1)
beta_dict,cf_dict,mse_train,mae_train,r_squared = GD_Linear(X_train3, X_test3, y_train, y_test,3)


# In[36]:


y_train.head()


# In[37]:


cf_final
mse_final 
r2_final
for alpha in [0.01]:
    cf_final = min(cf_dict[alpha])
    mse_final = min(mse_train[alpha])
    r2_final = max(r_squared[alpha])


# In[38]:


train_metrics3=pd.DataFrame([cf_final, mse_final,r2_final],index=['CF_Train','MSE_Train','R2_Train'],columns=['Random 10']).T

xt_test = X_test3.values.transpose()
rows_test = X_test3.shape[0]
cf_test = {}
mse_test = {}
r2_test = {}
for alpha in [0.01]:
    beta = beta_dict[alpha]
    pred_test = beta[0]+np.dot(beta[1:],xt_test)
    cf_test,mse_test,mae_test,r2_test = calculate_metrics(y_test,pred_test,rows_test)
    
test_metrics3 = pd.DataFrame([cf_test, mse_test,r2_test],index=['CF_Test','MSE_Test','R2_Test'],columns=['Random 10']).T
metrics3=pd.concat([train_metrics3,test_metrics3],axis=1)

full_model_metrics = metrics1.reindex(sorted(metrics1.columns), axis=1)
full_model_metrics = full_model_metrics.iloc[2:,[0,1,4,5,6,7]]
full_model_metrics=pd.DataFrame(full_model_metrics).T
full_model_metrics.columns=['Full Model']
full_model_metrics=pd.DataFrame(full_model_metrics).T

pd.concat([metrics3,full_model_metrics],axis=0)


# In[ ]:





# ## Logistic Regression

# ### Creating classes for Target Variable

# In[39]:


### Splitting classes based on its Median

df_logistic = df_filter.copy()
df_logistic['Appliances_class'] = [0 if x <= 60 else 1 for x in df_logistic['Appliances']] 
df_no_outliers=outlier(df_logistic)
df_logistic = df_logistic.drop(columns = ['Appliances'])


# In[40]:


df_logistic.head()


# In[41]:


df_logistic.shape


# In[42]:


pd.DataFrame(df_logistic['Appliances_class'].value_counts())


# ### Sigmoid Function

# In[43]:


def predict(beta,xt):
    z=beta[0]+(np.dot(beta[1:],xt))
    sigmoid = 1 / (1 + np.exp(-z))
    return(sigmoid)     


# ### Cost Calculation

# In[44]:


def cost_function(final_pred,y_train):
    a=y_train*np.log(final_pred)
    b=(1-y_train)*np.log(1-final_pred)
    cost=(a+b).sum()
    return(cost)


# ### Gradient Descent for Logistic Regression

# In[45]:


def GD_Logistic(X_train, X_test, y_train, y_test,exp):
    cf_dict={}
    beta_dict={}
    pred_proba_dict={}
    if exp == 1:
        lrate=[0.005,0.007,0.01]
    if exp == 3:
        lrate=[0.01]
    m=X_train.shape[0]
    n=X_train.shape[1]+1
    xt=X_train.values.transpose()

    for alpha in lrate:
        p_diff_cost=[0]*n
        beta=[0.05]*n
        cf=[0]*1000
        for iter in range(1000):
            for k in range(n):
                if(k == 0):
                    p_diff_cost[0]=(predict(beta,xt)-y_train).sum()
                else:
                    p_diff_cost[k]=np.dot((predict(beta,xt)-y_train),X_train.iloc[:,k-1])

            for bit in range(n):
                beta[bit]=round((beta[bit]-(alpha*p_diff_cost[bit])/m),4)

            pred_proba=predict(beta,xt)
            cf[iter]=-(cost_function(pred_proba,y_train))/m
        cf_dict[alpha]=cf
        beta_dict[alpha]=beta
        pred_proba_dict[alpha]=pred_proba
        
    return(cf_dict,beta_dict,pred_proba_dict)


# ### Experimentation 1: Varying Learning Rates

# In[46]:


df_logistic_scaled = feature_scaling(df_logistic)  ###Normalisation of features

x=df_logistic_scaled.iloc[:,:17]
y=df_logistic['Appliances_class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)
cf_dict,beta_dict,pred_proba_dict = GD_Logistic(X_train, X_test, y_train, y_test,1)


# In[47]:


beta_dict


# ### Cost Fuction Plot

# In[48]:


for i in [0.005,0.007,0.01]:
    plt.plot(cf_dict[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.xlabel("\nNumber of iterations")
plt.ylabel("\nCost Function")
plt.title("\nCost Function with different Learning rates\n")
plt.show()


# ### Values of Probabilities and Beta's for best Learning Rate

# In[49]:


pred_proba = pred_proba_dict[0.01]
beta=beta_dict[0.01]


# ### Predicting classes based on Probability results for Training data

# In[50]:


pred_class = [0 if i<=0.5 else 1 for i in pred_proba]
pred_class=pd.Series(pred_class)
pred_class.value_counts()


# ### Confusion Matrix for Training Data

# In[51]:


pd.DataFrame(confusion_matrix(y_train,pred_class),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[52]:


tn, fp, fn, tp = confusion_matrix(y_train,pred_class).ravel()
Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print(accuracy_score(y_train,pred_class))


# ### Test Dataset

# In[53]:


# xt=X_test.values.transpose()
# class_pred_test = predict(beta,xt)
# test_cf = -(cost_function(class_pred_test,y_test))/m
# print(test_cf)


# ### Predicting classes based on Probability results for Test data

# In[54]:


pred_test = [0 if i<=0.5 else 1 for i in class_pred_test]
pred_test=pd.Series(pred_test)


# ### Confusion Matrix for Test Dataset

# In[55]:


pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[56]:


tn, fp, fn, tp = confusion_matrix(y_test,pred_test).ravel()

Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print(accuracy_score(y_test,pred_test))


# In[57]:


fpr_train1, tpr_train1, thresholds_train1 = metrics.roc_curve(y_train, pred_proba,pos_label=1)
fpr_test1, tpr_test1, thresholds_test1 = metrics.roc_curve(y_test, class_pred_test,pos_label=1)
roc_auc_train1 = metrics.auc(fpr_train1, tpr_train1)
roc_auc_test1 = metrics.auc(fpr_test1, tpr_test1)
plt.plot(fpr_train1,tpr_train1,label='ROC curve (area = %0.2f) on training data' % roc_auc_train1)
plt.plot(fpr_test1,tpr_test1,label='ROC curve (area = %0.2f) on test data' % roc_auc_test1)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Training Data Vs Test Data ')
plt.legend(loc="lower right")
plt.show()


# ### Experimentation 3 

# In[58]:


random.seed(1)
random_features=random.sample(range(27), 10)

df=pd.read_csv("energydata_complete.csv")
df=df.drop(columns = ['date'])
df_no_outliers = outlier(df) ###Outlier Removal
df_scaled = feature_scaling(df_no_outliers)  ###Normalisation of features

df_no_outliers['Appliances_class'] = [0 if x <= 60 else 1 for x in df_no_outliers['Appliances']]
df_logistic = df_scaled.drop(columns = ['Appliances'])

x_random = df_logistic.iloc[:,random_features]
y=df_no_outliers['Appliances_class']

X_train3, X_test3, y_train, y_test = train_test_split(x_random, y, test_size=0.3,random_state=1)
cf_dict,beta_dict,pred_proba = GD_Logistic(X_train3, X_test3, y_train, y_test,3)


# In[59]:


X_train3.columns


# ### Exp 3 - Predicting the classes based on probabilties on Training Data 

# In[60]:


pred_class3 = [0 if i<=0.5 else 1 for i in pred_proba[0.01]]
pred_class3=pd.Series(pred_class3)
pred_class3.value_counts()


# ### Exp 3 - Confusion Matrix on Training Data

# In[61]:


pd.DataFrame(confusion_matrix(y_train,pred_class3),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[62]:


tn, fp, fn, tp = confusion_matrix(y_train,pred_class3).ravel()
Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print()
print(accuracy_score(y_train,pred_class3))


# ### Testing Data

# In[65]:


beta=beta_dict[0.01]
xt=X_test3.values.transpose()
class_pred_test = predict(beta,xt)
# test_cf = -(cost_function(class_pred_test,y_test))/m
# print(test_cf)

pred_test = [0 if i<=0.5 else 1 for i in class_pred_test]
pred_test=pd.Series(pred_test)

pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[66]:


tn, fp, fn, tp = confusion_matrix(y_test,pred_test).ravel()

Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print()
print(accuracy_score(y_test,pred_test))


# ### ROC Curves

# In[67]:


fpr_train3, tpr_train3, thresholds_train3 = metrics.roc_curve(y_train, pred_proba[0.01],pos_label=1)
roc_auc_train3 = metrics.auc(fpr_train3, tpr_train3)
plt.plot(fpr_train1,tpr_train1,label='ROC curve - Full Model (area = %0.2f)' % roc_auc_train1)
plt.plot(fpr_train3,tpr_train3,label='ROC curve - Random 10 (area = %0.2f)' % roc_auc_train3)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('\nROC Curve on Training Data with Full Model Vs Random 10 Model\n')
plt.legend(loc="lower right")
plt.show()


# In[68]:


fpr_test3, tpr_test3, thresholds_test3 = metrics.roc_curve(y_train, pred_proba[0.01],pos_label=1)
roc_auc_test3 = metrics.auc(fpr_test3, tpr_test3)
plt.plot(fpr_train1,tpr_train1,label='ROC curve - Full Model (area = %0.2f)' % roc_auc_test1)
plt.plot(fpr_test3,tpr_test3,label='ROC curve - Random 10 (area = %0.2f)' % roc_auc_test3)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('\nROC Curve on Test Data with Full Model and Random 10 Model\n')
plt.legend(loc="lower right")
plt.show()


# ### Experimentation 4 - Selecting the best 10 Features

# In[69]:


best_features=pd.DataFrame(df_no_outliers.corr()['Appliances']).abs()
best_features.sort_values(by=['Appliances'],ascending=False)


# In[70]:


df_no_outliers.head()


# In[71]:


df=pd.read_csv("energydata_complete.csv")
df_best_features = df.drop(columns =['date','lights','T3','T_out','rv1','rv2','RH_4','T4','T7','RH_7','T5','Press_mm_hg','RH_3','Tdewpoint','RH_5','Windspeed','RH_1','Visibility'])
df_best_no_outliers = outlier(df_best_features)
df_best_features = feature_scaling(df_best_no_outliers)

x=df_best_features.iloc[:,1:]
y=df_best_no_outliers['Appliances']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

beta_dict,cf_dict,mse_train,mae_train,r_squared = GD_Linear(X_train, X_test, y_train, y_test,4)


# In[72]:


X_train.head()


# In[73]:


for alpha in [0.01]:
    cf_final = min(cf_dict[alpha])
    mse_final = min(mse_train[alpha])
    r2_final = max(r_squared[alpha])


# In[74]:


train_metrics4=pd.DataFrame([cf_final, mse_final,r2_final],index=['CF_Train','MSE_Train','R2_Train'],columns=['Best 10']).T

xt_test = X_test.values.transpose()
rows_test = X_test.shape[0]
cf_test = {}
mse_test = {}
r2_test = {}
for alpha in [0.01]:
    beta = beta_dict[alpha]
    pred_test = beta[0]+np.dot(beta[1:],xt_test)
    cf_test,mse_test,mae_test,r2_test = calculate_metrics(y_test,pred_test,rows_test)
    
test_metrics4 = pd.DataFrame([cf_test, mse_test,r2_test],index=['CF_Test','MSE_Test','R2_Test'],columns=['Best 10']).T
metrics4=pd.concat([train_metrics4,test_metrics4],axis=1)

pd.concat([metrics3,metrics4,full_model_metrics],axis=0)


# ### Experimentation 4 - Logistic Regression

# In[ ]:


df=pd.read_csv("energydata_complete.csv")
df=df.drop(columns = ['date'])
df_no_outliers = outlier(df) ###Outlier Removal
df_scaled = feature_scaling(df_no_outliers)  ###Normalisation of features

df_no_outliers['Appliances_class'] = [0 if x <= 60 else 1 for x in df_no_outliers['Appliances']]
df_logistic = df_scaled.drop(columns = ['Appliances'])

x_best = df_logistic.drop(columns =['lights','T3','T_out','rv1','rv2','RH_4','T4','T7','RH_7','T5','Press_mm_hg','RH_3','Tdewpoint','RH_5','Windspeed','RH_1','Visibility'])
y=df_no_outliers['Appliances_class']

X_train, X_test, y_train, y_test = train_test_split(x_best, y, test_size=0.3,random_state=1)
cf_dict,beta_dict,pred_proba = GD_Logistic(X_train, X_test, y_train, y_test,3)


# In[1117]:


x_best.columns


# In[1090]:


pred_class4 = [0 if i<=0.5 else 1 for i in pred_proba[0.01]]
pred_class4=pd.Series(pred_class4)
pred_class4.value_counts()


# In[1091]:


pd.DataFrame(confusion_matrix(y_train,pred_class4),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[1092]:


tn, fp, fn, tp = confusion_matrix(y_train,pred_class4).ravel()
Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print()
print(accuracy_score(y_train,pred_class4))


# In[1093]:


beta=beta_dict[0.01]
xt=X_test.values.transpose()
class_pred_test = predict(beta,xt)
test_cf = -(cost_function(class_pred_test,y_test))/m
print(test_cf)

pred_test = [0 if i<=0.5 else 1 for i in class_pred_test]
pred_test=pd.Series(pred_test)

pd.DataFrame(confusion_matrix(y_test,pred_test),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1'])


# In[1094]:


tn, fp, fn, tp = confusion_matrix(y_test,pred_test).ravel()

Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)
print(Sensitivity)
print(Specificity)
print()
print(accuracy_score(y_test,pred_test))


# In[1099]:


fpr_train4, tpr_train4, thresholds_train4 = metrics.roc_curve(y_train, pred_proba[0.01],pos_label=1)
roc_auc_train4 = metrics.auc(fpr_train4, tpr_train4)
plt.plot(fpr_train1,tpr_train1,label='ROC curve - Full Model (area = %0.2f)' % roc_auc_train1)
plt.plot(fpr_train3,tpr_train3,label='ROC curve - Random 10 (area = %0.2f)' % roc_auc_train3)
plt.plot(fpr_train4,tpr_train4,label='ROC curve - Best 10 (area = %0.2f)' % roc_auc_train4)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('\nROC Curve on Training Data with with 3 models\n')
plt.legend(loc="lower right")
plt.show()


# In[1100]:


fpr_test4, tpr_test4, thresholds_test4 = metrics.roc_curve(y_train, pred_proba[0.01],pos_label=1)
roc_auc_test4 = metrics.auc(fpr_test4, tpr_test4)
plt.plot(fpr_train1,tpr_train1,label='ROC curve - Full Model (area = %0.2f)' % roc_auc_test1)
plt.plot(fpr_test3,tpr_test3,label='ROC curve - Random 10 (area = %0.2f)' % roc_auc_test3)
plt.plot(fpr_test4,tpr_test4,label='ROC curve - Best 10 (area = %0.2f)' % roc_auc_train4)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('\nROC Curve on Test Data with 3 Models \n')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




