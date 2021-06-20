#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Price Predictor


# In[2]:


import pandas as pd


# In[3]:


housing=pd.read_csv("C:/Users/HP/OneDrive/Desktop/ML/Book1.csv")


# In[4]:


housing.head()


# In[5]:


housing.info()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# In[ ]:





# In[10]:


#Train test splitting
import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


# train_set,test_set=split_train_test(housing,0.2)


# In[12]:


# print(f"Rows in train set: {len(train_set)}\n Rows in  test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\n Rows in  test set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


housing=strat_train_set.copy()


# In[17]:


strat_train_set['CHAS'].value_counts()


# In[18]:


#Looking for correlations

corr_matrix=housing.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[21]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[22]:


##Attribute combinations

housing["TAXRM"]=housing['TAX']/housing['RM']


# In[23]:


housing.head()


# In[24]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[25]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[26]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# In[27]:


# to take care of missing inputs
median=housing["RM"].median()


# In[28]:


median


# In[29]:


housing['RM'].fillna(median)


# In[30]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[31]:


imputer.statistics_


# In[32]:


X=imputer.transform(housing)


# In[33]:


housing_tr=pd.DataFrame(X,columns=housing.columns)
housing_tr.describe()


# In[34]:


# Scikit Learn Design

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),('std_scaler',StandardScaler()),])


# #Feature Scaling
# 
# 1. Min-max scaling- (normalization) -- (value-min)/(max-min)- scikit class called - MinMaxScaler
# 2. Standardization- (value-min)/std ------------------------- scykit class called - StandardScaler. It makes the variance as 1.

# In[35]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[36]:


housing_num_tr.shape


# In[37]:


#Selecting the desired model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[38]:


some_data=housing.iloc[:5]


# In[39]:


some_labels=housing_labels.iloc[:5]


# In[40]:


prepared_data=my_pipeline.transform(some_data)


# In[41]:


model.predict(prepared_data)


# In[42]:


list(some_labels)


# In[43]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
rmse


# Cross Validation

# In[44]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[45]:


rmse_scores


# In[46]:


def print_scores(scores):
    print("scores: ",scores)
    print("mean: ",scores.mean())
    print("Standard deviation: ",scores.std())


# In[47]:


print_scores(rmse_scores)


# In[48]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# In[54]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[55]:


final_mse


# In[57]:


prepared_data[0]


# Using the model

# In[58]:


from joblib import dump,load
import numpy as np
model=load('Dragon.joblib')

features=np.array([[-0.43225242,  3.48289015, -0.98372896, -0.27288841, -1.53213338,
       -0.32201178, -1.42736088,  2.94721287, -0.91790289, -0.59448701,
       -0.71805555,  0.39116709, -9.72112263]])
model.predict(features)


# In[ ]:




