#!/usr/bin/env python
# coding: utf-8

# In[15]:


# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#change working directory
import os
os.chdir(r'C:\Users\Evan Generoli\Downloads')


# In[17]:


# read in dataset
dat = pd.read_csv('HealthDat.csv', index_col=0)
dat.shape


# In[18]:


dat.info()


# In[19]:


dat.head()


# In[21]:


## make column names lower case
dat.columns = [x.lower() for x in dat.columns]
dat.head()
# alternate way
# dat.rename(str.lower, axis='columns')


# In[22]:


dat.columns


# In[8]:


# replace health (categorical) with dummy variables

right = pd.get_dummies(dat.health) # get dataframe of health category dummy variables
right.columns = [x + ' health' for x in right.columns]

left = dat.drop(columns='health') # get dataframe w/ health dropped

dat = left.join(right) # join two dataframes into one


dat.columns = [x.lower() for x in dat.columns] #make sure all column names are lower case
dat.head()


# In[9]:


# replace income (categorical) with dummy variables

right = pd.get_dummies(dat.income) # get dataframe of health category dummy variables

left = dat.drop(columns='income') # get dataframe w/ health dropped

dat = left.join(right) # join two dataframes into one


dat.columns = [x.lower() for x in dat.columns] #make sure all column names are lower case
dat.head()


# In[23]:


# drop year & birthmonth

dat = dat.drop(columns=['year','birthmo'])
dat.head()


# In[11]:


# data is good to go for EDA

dat.describe().round()


# In[14]:


dat.describe().round(2)


# In[60]:


dat.corr().round(2)


# In[268]:


dat[['white','black','asian']].corr().round(2)


# In[59]:


sns.heatmap(dat.corr().round(2), annot=True)


# In[ ]:


#### begin cluster analysis


# In[233]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(dat)


# In[234]:


kmeans.cluster_centers_


# In[257]:


x = kmeans.labels_
print(x)
print('length: ' + str(len(x)))


# In[247]:


kmeans.inertia_.round()


# In[259]:


inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(dat)
    inertia.append(kmeans.inertia_)


# In[260]:


index = list(range(1, 11))
plt.scatter(index, inertia)
plt.plot(index, inertia)
plt.ylabel('Within Cluster Sum of Squares')
plt.xlabel('Number of Clusters')
plt.title('Number of Clusters vs. Within Cluster Variability')


# In[ ]:





# In[ ]:


# import function for transforming categoricals into dummies

from sklearn.preprocessing import LabelEncoder()


# In[26]:


dat.head()


# In[ ]:


from mlxtend.plotting import category_scatter

fig = category_scatter(x='x', y='y', label_col='label', 
                       data=df, legend_loc='upper left')


# In[53]:


from mlxtend.plotting import category_scatter

fig = category_scatter(x='cigarette',y='age',label_col='income', data=dat, markers = 'o')


# In[58]:


color_codes = dat.health.astype('category').cat.codes
plt.scatter(x=dat.cigarette, y=dat.alcohol, c= color_codes, cmap='rainbow')

plt.ylabel('Alcohol')
plt.xlabel('Cigarettes')
plt.title('Alcohol vs. Cigarette Consumption by Health Status')

