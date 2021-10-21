#!/usr/bin/env python
# coding: utf-8

# # Case Study
# Below mentioned are the steps to be followed:
# 
# 1. Data Exploration and Understanding
#     * Load the Data
#     * Understand and Viz the data
#     * EDA
# 2. Data Preparation
#     * Removing Outliers and uneccessary columns
#     * Categorical variable treatment
#     * Standardizing numerical variables
#     * Train Test Split
# 3. Model Building and Training
#     * Use RFE for Feature Elimination
#     * Training the Model 
# 4. Prediction
# 5. Evaluation
#     * Use MSE, RMSE, MEA
#     * Use R2 score
# 6. Residual Analysis on Model
# 7. Principle Component Analysis
# 

# In[1]:


# Importing all necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# # 1. Data Exploration and Understanding

# In[3]:


df = pd.read_csv(r'C:\Users\welcome\Downloads\CO2 Emissions_Canada.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


# Data Description
df.info()


# In[7]:


df['Vehicle Class'].value_counts()


# In[8]:


df['Cylinders'] = df['Cylinders'].astype('category')
df['Cylinders'].value_counts()


# In[9]:


df['Transmission'] = df['Transmission'].astype('category')
df['Transmission'].value_counts()


# In[10]:


df['Gears'] = df['Transmission'].apply(lambda x:x[-1])
df['Gears'].value_counts()


# In[11]:


df['Gears'] = df['Gears'].replace('V','0')
df['Gears'].value_counts()


# In[12]:


df['Fuel Type'].value_counts()


# In[13]:


sns.boxplot(df['CO2 Emissions(g/km)'])


# In[14]:


Q1=df['CO2 Emissions(g/km)'].quantile(0.25)
Q3=df['CO2 Emissions(g/km)'].quantile(0.75)
IQR=Q3-Q1
Q3+1.5*IQR


# In[15]:


# Outlier Treatment
Q3=df.quantile(0.75)
IQR=Q3-Q1
df = df.loc[df['CO2 Emissions(g/km)']<= 408]


# In[16]:


df.shape


# In[17]:


# Hist Plots 
Numerical = df.select_dtypes(include = ['int64','float'])
Numerical.hist(figsize = (10,10))
plt.show()


# In[18]:


Categorical = df.select_dtypes(include=['category','object'])
Categorical.head()


# In[19]:


# Vehicle Class with CO2 Emission
plt.figure(figsize = (10,10))
sns.boxplot(data = df, x='Vehicle Class', y='CO2 Emissions(g/km)', palette = 'cubehelix')
plt.xticks(rotation = 90)
plt.show()


# In[20]:


# Gears with C02 Emission
plt.figure(figsize = (10,10))
sns.boxplot(data = df, x = 'Gears', y = 'CO2 Emissions(g/km)', palette = 'vlag')
plt.show()


# In[21]:


# Car Makers with C02 Emission
plt.figure(figsize = (20,15))
sns.boxplot(data = df, x='Make', y='CO2 Emissions(g/km)', palette = 'husl')
plt.xticks(rotation = 90)
plt.show()


# In[22]:


# City Fuel Consumption vs Highway Fuel Consumption with Fuel Category
plt.figure(figsize = (10,10))
sns.scatterplot(data = df, x = 'Fuel Consumption City (L/100 km)', y = 'Fuel Consumption Hwy (L/100 km)', hue = 'Fuel Type')
plt.show()


# In[23]:


# Fuel Consumption by Fuel Categories
df.groupby(by = 'Fuel Type')['Fuel Consumption Comb (L/100 km)'].mean()


# In[24]:


# Pivot Table with Cylinders, Fuel Type and C02 Emissions
df.pivot_table(values = ['CO2 Emissions(g/km)'], index = ['Cylinders','Fuel Type'], aggfunc = 'mean')


# In[25]:


# Pairplot for all values
sns.pairplot(df, palette = 'husl')


# In[26]:


# Heatmap for all vvalues
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# From Heatmap, we get some interesting points :
# 1. All Numerical Values are highly correlated to C02 Emission.
# 2. There is huge correlation between independent variables, which leads to interpretibility problem. 

# In[27]:


df.head()


# # 2. Data Preparation 

# In[28]:


# Standardization for Numerical Data
from sklearn.preprocessing import StandardScaler
df_num = pd.DataFrame(StandardScaler().fit_transform(Numerical), 
                      columns = Numerical.columns)
df_num.head()


# In[29]:


Categorical = Categorical.loc[:,['Vehicle Class','Transmission','Fuel Type','Cylinders','Gears']]
Categorical.head()


# In[30]:


# One hot Encoding for Categorical Values
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)

df_cat = pd.DataFrame(encoder.fit_transform(Categorical), columns=encoder.get_feature_names(Categorical.columns))
df_cat


# In[31]:


print(df_cat.shape)
print(df_num.shape)


# In[32]:


data = pd.concat([df_cat, df_num], axis = 1)
data.head()


# In[33]:


target = data.pop('CO2 Emissions(g/km)')
data = data


# In[34]:


print(data.shape)
print(target.shape)


# In[35]:


# Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:





# # 3. Model Building and Training

# In[235]:


# Feature Elimination using RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
LR = LinearRegression()
rfe = RFE(LR, n_features_to_select=10, step=1)
rfe = rfe.fit(X_train, y_train)


# In[236]:


temp_df = pd.DataFrame({'Columns': X_train.columns, 'Included': rfe.support_, 'Ranking': rfe.ranking_})
temp_df.loc[temp_df.Included == True, :]


# In[237]:


X_train_cols = list(X_train.columns[rfe.support_])


# In[238]:


X_train_rfe = X_train[X_train_cols]

X_train_rfe.shape


# In[239]:


X_train_rfe.head()


# In[240]:


plt.figure(figsize=(18, 9))
sns.heatmap(X_train_rfe.corr(), cmap="YlGnBu")


# In[241]:


X_train_rfe = X_train_rfe.drop(columns=['Fuel Type_E', 'Fuel Type_N','Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)'])


# We are dropping some columns due to multicollinearity issue

# In[242]:


rfe.fit(np.array(X_train_rfe), y_train)


# In[243]:


X_test_rfe = X_test[X_train_rfe.columns]


# # 4. Prediction

# In[244]:


y_test_pred_rfe = rfe.predict(np.array(X_test_rfe))


# # 5. Evaluation

# In[245]:


# Metrics for Performance Checking
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_test_pred_rfe))
print(metrics.mean_squared_error(y_test, y_test_pred_rfe))
print(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_rfe)))


# In[246]:


# R2 score
print(metrics.r2_score(y_test, y_test_pred_rfe))


# In[247]:


temp_df =pd.DataFrame({'y_test':y_test, 'y_test_pred_rfe': y_test_pred_rfe})
temp_df.tail()


# # 6. Residual Analysis of Model

# In[109]:


plt.figure(figsize = (10,10))
y_train_pred = LR.predict(X_train_rfe)
Residual = y_train - y_train_pred
sns.distplot(Residual, color='green', kde=True)
plt.show()


# It shows normal distribution ~(0, std)

# In[110]:


plt.scatter(Residual, y_train)


# 1. It follows homoscadisticity
# 2. Residual and training data are independent of each other

# # 7. Principal Component Analysis

# In[111]:


X_train.shape


# In[112]:


X_train.head()


# In[113]:


from sklearn.decomposition import PCA
pca = PCA(random_state= 0)
pca.fit(X_train)


# In[114]:


plt.figure(figsize = (15,15))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('No of Components')
plt.ylabel('No of Components')
plt.show()


# In[115]:


# Cummulative Sum of Eigen Vectors
np.cumsum(pca.explained_variance_ratio_)


# In[116]:


from sklearn.decomposition import PCA

pca_final = PCA(n_components = 14, random_state = 0)

X_train_pca = pca_final.fit_transform(X_train)


# In[117]:


X_train_pca.shape


# In[118]:


# Correlation matrix between Eigen vector of PCA
corrmat = np.corrcoef(X_train_pca.T)
plt.figure(figsize=(18, 9))
sns.heatmap(corrmat, cmap="YlGnBu")


# In[119]:


X_test_pca = pca_final.transform(X_test)


# In[120]:


LR.fit(X_train_pca, y_train)
y_test_pred_pca = LR.predict(X_test_pca)


# In[121]:


# Metrics for Performance Checking
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_test_pred_pca))
print(metrics.mean_squared_error(y_test, y_test_pred_pca))
print(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_pca)))


# In[122]:


print(metrics.r2_score(y_test, y_test_pred_pca))


# In[123]:


temp_df =pd.DataFrame({'y_test':y_test, 'y_test_pred_pca': y_test_pred_pca})
temp_df.tail()


# In[124]:


plt.figure(figsize = (10,10))
y_train_pca = LR.predict(X_train_pca)
Residual_pca = y_train - y_train_pca
sns.distplot(Residual_pca, color='green', kde=True)


# In[219]:


