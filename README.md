# EDA_Wine_Quality_Data_Red

# %% [markdown]
# ## EDA With Red Wine Data
# 
# Data Set Information:
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# 
# Attribute Information:
# 
# Input variables (based on physicochemical tests):
# - 1 - fixed acidity
# - 2 - volatile acidity
# - 3 - citric acid
# - 4 - residual sugar
# - 5 - chlorides
# - 6 - free sulfur dioxide
# - 7 - total sulfur dioxide
# - 8 - density
# - 9 - pH
# - 10 - sulphates
# - 11 - alcohol
# 
# Output variable (based on sensory data):
# - 12 - quality (score between 0 and 10)

#
import pandas as pd
df=pd.read_csv('winequality-red.csv',sep=';')
df.head()


<img width="1414" height="215" alt="image" src="https://github.com/user-attachments/assets/a340b282-8ebc-497d-8a72-86219fa7486c" />

# %%
# Descriptive Summary of the dataset


df.describe()
<img width="1743" height="321" alt="image" src="https://github.com/user-attachments/assets/6a176b95-9ae0-47e8-b3da-9378aed7543c" />

# 
#Summary of the dataset


df.info()
<img width="578" height="468" alt="image" src="https://github.com/user-attachments/assets/e7b2a274-fd6c-4f98-a244-e8820c2b6d06" />

# %%
#Shape of the Dataset


df.shape

# %%
#List down all column names


df.columns

# %%
#number of UNique values in Target Variable


df['quality'].unique()

# %%
#Missing Values in the Dataset


df.isnull().sum()

# %%
#Check for Duplicate records


df[df.duplicated()]

# %%
#Remove Duplicates
df.drop_duplicates(inplace=True)

# %%
df.shape


# %%
#Correlation

df.corr()

# %%
#Plotting Correlation 


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
<img width="1138" height="802" alt="image" src="https://github.com/user-attachments/assets/1184c934-8727-4fe1-a3b6-2b0250c3a7a4" />

# %%
#Visulaizing the Target Variable


df.quality.value_counts().plot(kind='bar')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.show()
<img width="757" height="564" alt="image" src="https://github.com/user-attachments/assets/2795a143-a132-4814-ad88-8e221b5f798f" />

## Conclusion : The Dataset is Imbalanced, since Wine Quality 5 & 6 have maximum datapoints compared to other quality Score.

# %%
#Univariate,Bivariate,Multivariate Analysis


sns.pairplot(df)
<img width="1753" height="903" alt="image" src="https://github.com/user-attachments/assets/80d0871d-b2d3-490f-864d-fb5d1d769041" />

# %%
#Categorical Plot


sns.catplot(x='quality',y='alcohol',data=df,kind = 'box')
# the below image shows Quality score of 5 has more outliers than any other Quality
<img width="720" height="654" alt="image" src="https://github.com/user-attachments/assets/f06f07e5-96b1-4869-b4fa-9d9c4cc3808f" />

# %%
#Scatter plot between to Features 


sns.scatterplot(x='alcohol',y='pH',hue='quality',data=df)
<img width="746" height="559" alt="image" src="https://github.com/user-attachments/assets/2748ef4c-05b6-4a29-a1c3-3f8e67c7060d" />

# %%



