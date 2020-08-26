#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[79]:


df = pd.read_csv('C:/Users/Hp/Downloads/diabetes.csv')
df.head()


# In[80]:


df.drop_duplicates()


# In[81]:


df.shape


# In[82]:


df.isnull().sum()


# In[83]:


df.describe()


# # DATA ANALYSIS

# In[84]:


sns.countplot(df['Outcome'])


# In[104]:


sns.countplot(df['Pregnancies'])


# In[86]:


SkinThicknessa=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']


# In[87]:


for i in a:
    sns.distplot(df[i])
    plt.show()


# In[89]:


sns.countplot(df['Pregnancies'],hue=df['Outcome'])


# In[101]:


sns.swarmplot(y=df['Pregnancies'],x=df['Outcome'])


# In[116]:


sns.barplot(y=df['BMI'],x=df['Outcome'])


# In[114]:


sns.barplot(y=df['Insulin'],x=df['Outcome'])


# In[115]:


sns.barplot(y=df['SkinThickness'],x=df['Outcome'])


# In[106]:


sns.swarmplot(y=df['Age'],x=df['Outcome'])


# In[99]:


sns.scatterplot(x=df['BMI'],y=df['Age'],hue=df['Outcome'])


# In[77]:


sns.scatterplot(x=df['Pregnancies'],y=df['Age'],hue=df['Outcome'])


# In[111]:


sns.scatterplot(x=df['Glucose'],y=df['Age'],hue=df['Outcome'])


# In[108]:


sns.scatterplot(x=df['BloodPressure'],y=df['Age'],hue=df['Outcome'])


# # ANALYSIS REPORT
1. Patients aged between 21-38 have diabetes more in number.
2. Patients below 22 BMI do not have diabetes comparitively.
3. Patients with 0 number of pregnancies suffer diabetes more comparitively.
4. Patients with diabetes have Glucose level more than 100.
5. Patients with Insulin level above 70 have diabetes.
6. Patients with Skin Thickness above 20 have diabetes.
7. Patients with BMI above 30 have diabetes.
# In[254]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix


# In[264]:


x=df[['BMI','Age','Pregnancies']]
#x=df.drop('Outcome',axis=1)
y=df[['Outcome']]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)# 80% training data


# In[265]:


xtrain.shape,ytrain.shape


# In[266]:


xtest.shape,ytest.shape


# In[267]:


model=DecisionTreeClassifier(criterion='entropy',max_depth=4)
model.fit(xtrain,ytrain)


# In[268]:


plt.figure(figsize=(20,10))
plot_tree(model,filled=True)
plt.show()


# In[271]:


#print(confusion_matrix(ytest,ypred))
ytrain_pred=model.predict(xtrain)
ypred=model.predict(xtest)
print('accuracy = ',accuracy_score(ytest,ypred))
#ypred.shape,ytrain_pred.shape


# In[ ]:




