#!/usr/bin/env python
# coding: utf-8

# # <u>***BANK CUSTOMER CHURN PREDICTION***</u>
# 
# **Problem : Predicting customer churn in banking industry using machine learning algorithms.**

# **Reminder!!!** 
# * **I intervened in the code for some visualizations that appeared as blank plots in the Jupyter notebook, but it did not make a difference. If my visuals are still inaccessible, you can try changing the notebook from "Not Trusted" to "Trusted". This will solve the problem.Thank you for your understanding.**

# # Table of Contents
# 
# * **[1. SUMMARY](#1.-SUMMARY)**
# * **[2. DATA](#2.-DATA)**
# * > [2.1 About the Features](#2.1-About-the-Features)
# * > [2.2 Problem](#2.2-Problem )
# * > [2.3 Target Variable](#2.3-Target-Variable)
# * **[3. ANALYSIS](#3.-ANALYSIS)**
# * > [3.1 Reading the Data](#3.1-Reading-the-Data)
# * > [3.2 Exploratory Data Anaysis (EDA) & Visulization](#3.2-Exploratory-Data-Anaysis-(EDA)-&-Visulization)
# * >> [3.2.1 The Examination of Target Variable](#3.2.1-The-Examination-of-Target-Variable)
# * >> [3.2.2 The Examination of Numerical Features](3.2.2-The-Examination-of-Numerical-Features)
# * >>> [3.2.2.1 The Examination of Skewness & Kurtosis](3.2.2.1-The-Examination-of-Skewness-&-Kurtosis)
# * >> [3.2.3 The Examination of Categorical Features](#3.2.3-The-Examination-of-Categorical-Features)
# * >>> [3.2.3.1 Surname and Exited](#3.2.3.1-Surname-and-Exited)
# * >>> [3.2.3.2 Geography and Exited](#3.2.3.2-Geography-and-Exited)
# * >>> [3.2.3.3 Gender and Exited](#3.2.3.3-Gender-and-Exited)
# * >> [3.2.4 Dummy Variables Operation](#3.2.4-Dummy-Variables-Operation)
# * **[4. TRAIN - TEST SPLIT & HANDLING WITH MISSING VALUES](#4.-TRAIN---TEST-SPLIT-&-HANDLING-WITH-MISSING-VALUES)**
# * > [4.1 Train - Test Split](#4.1-Train---Test-Split)
# * > [4.2 Handling with Missing Values](#4.2-Handling-with-Missing-Values)
# * **[5. FEATURE SCALING](#5.-FEATURE-SCALING)**
# * > [5.1 The Implementation of Scaling](#5.1-The-Implementation-of-Scaling)
# * > [5.2 General Insights Before Going Further](#5.2-General-Insights-Before-Going-Further)
# * > [5.3 Handling with Skewness with PowerTransform & Checking Model Accuracy Scores](#5.3-Handling-with-Skewness-with-PowerTransform-&-Checking-Model-Accuracy-Scores)
# * **[6. MODELLING & MODEL PERFORMANCE](#6.-MODELLING-&-MODEL-PERFORMANCE)**
# * > [6.1 The Implementation of Logistic Regression (LR)](#6.1-The-Implementation-of-Logistic-Regression-(LR))
# * >> [6.1.1 Modelling Logistic Regression (LR) with Default Parameters](#6.1.1-Modelling-Logistic-Regression-(LR)-with-Default-Parameters)
# * >> [6.1.2 Cross-Validating Logistic Regression (LR) Model](#6.1.2-Cross-Validating-Logistic-Regression-(LR)-Model)
# * >> [6.1.3 Modelling Logistic Regression (LR) with Best Parameters Using GridSeachCV](#6.1.3-Modelling-Logistic-Regression-(LR)-with-Best-Parameters-Using-GridSeachCV)
# * >> [6.1.4 ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#6.1.4-ROC-(Receiver-Operating-Curve)-and-AUC-(Area-Under-Curve))
# * >> [6.1.5 The Determination of The Optimal Threshold](#6.1.5-The-Determination-of-The-Optimal-Threshold)
# * > [6.2 The Implementation of Support Vector Machine (SVM)](#6.2-The-Implementation-of-Support-Vector-Machine-(SVM))
# * >> [6.2.1 Modelling Support Vector Machine (SVM) with Default Parameters](#6.2.1-Modelling-Support-Vector-Machine-(SVM)-with-Default-Parameters)
# * >> [6.2.2 Cross-Validating Support Vector Machine (SVM) Model](#6.2.2-Cross-Validating-Support-Vector-Machine-(SVM)-Model)
# * >> [6.2.3 Modelling Support Vector Machine (SVM) with Best Parameters Using GridSeachCV](#6.2.3-Modelling-Support-Vector-Machine-(SVM)-with-Best-Parameters-Using-GridSeachCV)
# * >> [6.2.4 ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#6.2.4-ROC-(Receiver-Operating-Curve)-and-AUC-(Area-Under-Curve))
# * > [6.3 The Implementation of K-Nearest Neighbor (KNN)](#6.3-The-Implementation-of-K-Nearest-Neighbor-(KNN))
# * >> [6.3.1 Modelling K-Nearest Neighbor (KNN) with Default Parameters](#6.3.1-Modelling-K-Nearest-Neighbor-(KNN)-with-Default-Parameters)
# * >> [6.3.2 Cross-Validating K-Nearest Neighbor (KNN)](#6.3.2-Cross-Validating-K-Nearest-Neighbor-(KNN))
# * >> [6.3.3 Elbow Method for Choosing Reasonable K Values](#6.3.3-Elbow-Method-for-Choosing-Reasonable-K-Values)
# * >> [6.3.4 GridsearchCV for Choosing Reasonable K Values](#6.3.4-GridsearchCV-for-Choosing-Reasonable-K-Values)
# * >> [6.3.5 ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#6.3.5-ROC-(Receiver-Operating-Curve)-and-AUC-(Area-Under-Curve))
# * > [6.4 The Implementation of GradientBoosting (GB)](#6.4-The-Implementation-of-GradientBoosting-(GB))
# * >> [6.4.1 Modelling GradientBoosting (GB) with Default Parameters](#6.4.1-Modelling-GradientBoosting-(GB)-with-Default-Parameters)
# * >> [6.4.2 Cross-Validating GradientBoosting (GB)](#6.4.2-Cross-Validating-GradientBoosting-(GB))
# * >> [6.4.3 Feature Importance for GradientBoosting (GB) Model](#6.4.3-Feature-Importance-for-GradientBoosting-(GB)-Model)
# * >> [6.4.4 Modelling GradientBoosting (GB) Model with Best Parameters Using GridSeachCV](#6.4.4-Modelling-GradientBoosting-(GB)-Model-with-Best-Parameters-Using-GridSeachCV)
# * >> [6.4.5 ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#6.4.5-ROC-(Receiver-Operating-Curve)-and-AUC-(Area-Under-Curve))
# * **[7. THE COMPARISON OF MODELS](#7.-THE-COMPARISON-OF-MODELS)**
# * **[8. CONCLUSION](#8.-CONCLUSION)**
# * **[9. REFERENCES](#9.-REFERENCES)**
# * > [Note.1](#Note.1)
# * > [Note.2](#Note.2)
# * > [Note.3](#Note.3)

# # 1. SUMMARY 
# 
# I employed Exploratory Data Analysis (EDA) and various Model Classifications, including Logistic Regression (LR), Support Vector Machine (SVM),  K-Nearest Neighbors (KNN) and Gradient Boosting (GB) to examine the dataset 'Bank Customer Churn Prediction' from the Kaggle website, which is labeled as 'Churn_Modelling.csv.' (https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction?resource=download ).
# 
# I attempted to explore the dataset comprehensively, examining various aspects and visualizing as much as possible to gain insights into the data. I employed four(4) Machine Learning Classification Algorithms.

# In[1]:


import time

def elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}"
                .format(int(hours),int(minutes),seconds))



# First cell of my notebook 
start = time.time()


# In[2]:


# installing the necessary libraries

get_ipython().system('pip3 install pyforest')
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[3]:


# importing necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import make_scorer, precision_score, PrecisionRecallDisplay, RocCurveDisplay, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')

# importing plotly and cufflinks in offline mode

import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('max_colwidth',200)

pd.set_option('display.max_rows', 100) 
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import colorama
from colorama import Fore, Style  
# makes strings colored
get_ipython().system('pip3 install termcolor')
from termcolor import colored


# In[4]:


# Function for determining the number and percentages of missing values

def missing (df):
    num_of_missing = df.isnull().sum().sort_values(ascending=False)
    percentage_of_missing = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([num_of_missing, percentage_of_missing], axis=1, keys=['Num_of_missing', 'Percentage_of_missing'])
    return missing_values


# In[5]:


# Function for insighting summary information about the column

def first_looking(col):
    print("column name    : ", col)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col].isnull().sum()/df.shape[0]*100, 2))
    print("num_of_nulls   : ", df[col].isnull().sum())
    print("num_of_uniques : ", df[col].nunique())
    print(df[col].value_counts(dropna = False))


# In[6]:


# Function for examining scores

def train_val(y_train, y_train_pred, y_test, y_pred):
    
    scores = {"train_set": {"Accuracy" : accuracy_score(y_train, y_train_pred),
                            "Precision" : precision_score(y_train, y_train_pred),
                            "Recall" : recall_score(y_train, y_train_pred),                          
                            "f1" : f1_score(y_train, y_train_pred)},
    
              "test_set": {"Accuracy" : accuracy_score(y_test, y_pred),
                           "Precision" : precision_score(y_test, y_pred),
                           "Recall" : recall_score(y_test, y_pred),                          
                           "f1" : f1_score(y_test, y_pred)}}
    
    return pd.DataFrame(scores)


# # 2. DATA
# 
# ## 2.1 About the Features
# 
# The bank customer churn dataset is a commonly used dataset for predicting customer churn in the banking industry. It contains information on bank customers who either left the bank or continue to be a customer. The dataset includes the following attributes:
# 
# * Customer ID: A unique identifier for each customer
# * Surname: The customer's surname or last name
# * Credit Score: A numerical value representing the customer's credit score
# * Geography: The country where the customer resides (France, Spain or Germany)
# * Gender: The customer's gender (Male or Female)
# * Age: The customer's age.
# * Tenure: The number of years the customer has been with the bank
# * Balance: The customer's account balance
# * NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
# * HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
# * IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
# * EstimatedSalary: The estimated salary of the customer
# * Exited: Whether the customer has churned (1 = yes, 0 = no)

# ## 2.2 Problem
# 
# This is a binary classification problem where I will make predictions on the target variable "Exited". Subsequently, I will compare the predictions of four machine learning algorithms and attempt to determine the best-performing model.

# ## 2.3 Target Variable
# 
# In machine learning, the target variable, also known as the dependent variable or response variable, is the variable that you are trying to predict or explain. It is the outcome variable of interest. The goal of a machine learning model is often to learn a mapping from the input features to the target variable.
# 
# For example, if you are working on a binary classification problem to predict whether an email is spam or not, the target variable would be binary, indicating either spam (1) or not spam (0). In a regression problem, where you're predicting a numerical value, the target variable might be a continuous variable like the price of a house.
# 
# In my study, the target variable is "Exited", where I aim to determine whether someone is likely to churn based on input parameters such as age, balance, tenure, and various test results.

# # 3. ANALYSIS

# ## 3.1 Reading the Data

# In[7]:


df = pd.read_csv('Churn_Modelling.csv', index_col ="CustomerId")


# In[8]:


df.drop("RowNumber", axis = 1, inplace = True)


# ## 3.2 Exploratory Data Anaysis (EDA) & Visulization

# In[9]:


df


# In[10]:


df.head()


# In[11]:


df.tail(10)


# In[12]:


df.duplicated().value_counts()


# In[13]:


df = df.drop_duplicates()
df.duplicated().value_counts()


# In[14]:


df.sample(10)


# In[15]:


df.columns


# In[16]:


print("My dataset consists of ", df.shape[0], "rows and", df.shape[1], "columns")


# In[17]:


df.info()


# In[18]:


df.describe().T


# In[19]:


df.describe(include=object).T


# In[20]:


df.nunique()


# In[21]:


# to find how many unique values object features have

for col in df.select_dtypes(include=[np.number]).columns:
  print(f"{col} has {df[col].nunique()} unique value")


# In[22]:


df.duplicated().value_counts()


# In[23]:


missing (df)


# In[24]:


print(df[df['Geography'].isnull()]) #customerID=15592531


# In[25]:


print(df[df['Age'].isnull()]) #customerID=15592389


# In[26]:


print(df[df['HasCrCard'].isnull()]) #customerID=15737888


# In[27]:


print(df[df['IsActiveMember'].isnull()]) #customerID=15792365


# In[28]:


df['Geography'].fillna('France', inplace = True)


# In[29]:


df['Age'].fillna(df['Age'].mean(), inplace = True)


# In[30]:


print(df[df['Age'].isnull()]) #customerID=15592389


# In[31]:


df[9:10]


# In[32]:


df['HasCrCard'].fillna(1.000, inplace = True)
df['IsActiveMember'].fillna(0.000, inplace = True)


# In[33]:


missing (df)


# ### 3.2.1 The Examination of Target Variable

# In[34]:


first_looking("Exited")


# In[35]:


print(df["Exited"].value_counts())
df["Exited"].value_counts().plot(kind="pie", autopct='%1.1f%%', figsize=(10,10));


# In[36]:


# discovering the numbers and percentages of churned and NOT churned customers

y = df['Exited']
print(f'Percentage of Churned Customer: % {round(y.value_counts(normalize=True)[1]*100,2)} --> \
({y.value_counts()[1]} cases for Churned Customer)\nPercentage of NOT Churned Customer: % {round(y.value_counts(normalize=True)[0]*100,2)} --> ({y.value_counts()[0]} cases for NOT Churned Customer)')


# In[37]:


df['Exited'].describe()


# In[38]:


# discovering the statistical values over other variables of NOT churned customers 

df[df['Exited']==0].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[39]:


# discovering the statistical values over other variables of churned customers 

df[df['Exited']==1].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[40]:


print( f"Skewness: {df['Exited'].skew()}")


# In[41]:


# it indicates that the distribution of the 'Exited' column is positively skewed. 
# this means that the right tail of the distribution is longer or fatter than the left tail.


# In[42]:


print( f"Kurtosis: {df['Exited'].kurtosis()}")


# In[43]:


# the distribution of the 'Exited' column in the DataFrame has a positive but small kurtosis value (0.16567104336407112), 
#it suggests a slightly heavier tail than a normal distribution, but not significantly so.


# In[44]:


df['Exited'].iplot(kind='hist')


# In[45]:


# Spliting Dataset into numeric & categoric features

numerical= df.drop(['Exited'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')


# ### 3.2.2 The Examination of Numerical Features

# In[46]:


df[numerical].head().T


# In[47]:


df[numerical].describe().T


# In[48]:


df[numerical].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[49]:


df[numerical].iplot(kind='hist');


# In[50]:


df[numerical].iplot(kind='histogram', subplots=True,bins=50)


# In[51]:


for i in numerical:
    df[i].iplot(kind="box", title=i, boxpoints="all", color='lightseagreen')


# In[52]:


index = 0
plt.figure(figsize=(20,20))
for feature in numerical:
    if feature != "Exited":
        index += 1
        plt.subplot(4, 3, index)
        sns.boxplot(x='Exited', y=feature, data=df)


# In[53]:


fig = px.scatter_3d(df, 
                    x='CreditScore',
                    y='Age',
                    z='HasCrCard',
                    color='Exited')
fig.show();


# In[54]:


sns.pairplot(df, hue="Exited", palette="inferno", corner=True);


# ### 3.2.2.1 The Examination of Skewness & Kurtosis

# In[55]:


skew_vals = df.skew().sort_values(ascending=False)
skew_vals


# In[56]:


skew_limit = 0.5 # This is our threshold-limit to evaluate skewness. Overall below abs(1) seems acceptable for the linear models. 
skew_vals = df.skew()
skew_cols = skew_vals[abs(skew_vals)> skew_limit].sort_values(ascending=False)
skew_cols 


# In[57]:


#Interpreting Skewness 

for skew in skew_vals:
    if -0.5 < skew < 0.5:
        print ("A skewness value of", '\033[1m', Fore.GREEN, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.GREEN, "symmetric", '\033[0m')
    elif  -0.5 < skew < -1.0 or 0.5 < skew < 1.0:
        print ("A skewness value of", '\033[1m', Fore.YELLOW, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.YELLOW, "moderately skewed", '\033[0m')
    else:
        print ("A skewness value of", '\033[1m', Fore.RED, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.RED, "highly skewed", '\033[0m')


# # Note.1
# 
# There are three types of kurtosis:
# 
# 1.  Mesokurtic: This occurs when the tails of the distribution are similar to a normal distribution, and the kurtosis value is It indicates a bell-shaped curve with tails that are neither too heavy nor too light.
# 
# 2.  Leptokurtic: If the kurtosis is greater than 3, the distribution is leptokurtic. This means that the tails are heavier than a normal distribution, suggesting the presence of many outliers. The shape is recognized as a thin, high-peaked bell curve.
# 
# 3.  Platykurtic: If the kurtosis is less than 3, the distribution is platykurtic. This implies thinner tails or a lack of outliers compared to a normal distribution. In a platykurtic distribution, the bell-shaped curve is broader, and the peak is lower than in a mesokurtic distribution.
# 
# The data is considered to be normal if skewness is between -2 and +2, and kurtosis is between -7 and +7. Additionally, multi-normality data tests involve checking for skewness less than 3, kurtosis between -2 and 2, and meeting the Mardia criterion (less than 3).
# 
# Finally, the use of skewness and kurtosis indices is discussed for identifying the normality of data. 

# In[58]:


kurtosis_vals = df.kurtosis().sort_values(ascending=False)
kurtosis_vals


# In[59]:


# Calculating Kurtosis 

kurtosis_limit = 7 # This is our threshold-limit to evaluate skewness. Overall below abs(1) seems acceptable for the linear models.
kurtosis_vals = df.kurtosis()
kurtosis_cols = kurtosis_vals[abs(kurtosis_vals) > kurtosis_limit].sort_values(ascending=False)
kurtosis_cols


# To prevent data leakage, I need to address kurtosis and skewness issues after splitting my data into train and test sets. For this purpose, I will use pipeline(), as the pipeline can be employed as any other estimator and avoids leaking the test set into the train set.

# In[60]:


# the correlation among variables using heatmap

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)
plt.xticks(rotation=45);


# In[61]:


# multicollinearity check among variables

df_temp = df.corr()

count = "Done"
feature =[]
collinear=[]
for col in df_temp.columns:
    for i in df_temp.index:
        if (df_temp[col][i]> .9 and df_temp[col][i] < 1) or (df_temp[col][i]< -.9 and df_temp[col][i] > -1) :
                feature.append(col)
                collinear.append(i)
                print(Fore.RED + f"\033[1mmulticolinearity alert in between\033[0m {col} - {i}")
        else:
            print(f"For {col} and {i}, there is NO multicollinearity problem") 

print("\033[1mThe number of strong corelated features:\033[0m", count) 


# ### 3.2.3 The Examination of Categorical Features

# In[62]:


df[categorical].head().T


# In[63]:


df[categorical].describe()


# In[64]:


for i in categorical:
    df[i].iplot(kind="box", title=i, boxpoints="all", color='lightseagreen')


# In[65]:


df[categorical].iplot(kind='hist');


# In[66]:


df[categorical].iplot(kind='histogram',subplots=True,bins=50)


# ### 3.2.3.1 Surname and Exited

# In[67]:


df["Surname"].value_counts()


# In[68]:


df['Surname'].iplot(kind='hist', )


# In[69]:


sns.swarmplot(y="Surname", x="Gender", hue="Exited", data=df, palette="husl");


# ### 3.2.3.2 Geography and Exited

# In[70]:


df["Geography"].value_counts()


# In[71]:


df['Geography'].iplot(kind='hist', )


# In[72]:


#sns.swarmplot(y="Gender", x="Geography", hue="Exited", data=df, palette="husl");


# ### 3.2.3.3 Gender and Exited

# In[73]:


df["Gender"].value_counts()


# In[74]:


df['Gender'].iplot(kind='hist')


# In[75]:


sns.catplot(
    data=df, kind="swarm",
    x="Gender", y="Surname", hue="Exited", col="Geography",
    aspect=.5
)


# ### 3.2.4 Dummy Variables Operation
# 
# A dummy variable is a binary variable, taking values of 0 and 1, to represent the presence or absence of a particular condition or characteristic. For instance, in a medical study, a dummy variable might be used to differentiate between a placebo (0) and a drug (1).
# 
# When dealing with a categorical variable that has more than two categories, a set of dummy variables can be employed. Each dummy variable corresponds to a specific category, and their values indicate the presence or absence of that category. For example, a variable like "Region" with categories "North," "South," "East," and "West" could be represented by dummy variables such as "North_dummy," "South_dummy," "East_dummy," and "West_dummy."
# 
# Numeric variables can also be dummy coded to explore nonlinear effects. For instance, if investigating the impact of age on a certain outcome, dummy variables for age groups ("Young," "Middle-aged," "Senior") could be created to capture potential nonlinear relationships.
# 
# Dummy variables are known by various names, including:
# 
# * Indicator Variables: These variables indicate the presence or absence of a specific condition.
# * Design Variables: Used in experimental design, representing different experimental conditions.
# * Contrasts: Applied in statistical analyses to represent differences between groups or conditions.
# * One-Hot Coding: A term commonly used in machine learning, especially when converting categorical variables into a format suitable for algorithms.
# * Binary Basis Variables: These variables serve as a basis for binary comparisons, often used in contrasts between different categories.

# In[76]:


df.shape


# In[77]:


df.head()


# In[78]:


df[categorical].value_counts()


# In[79]:


# because 'Surname' is not a logical variable for analysis, I have removed it from my dataframe before test-train data split

df2=df.copy(deep=True) 
df2.drop(['Surname'], axis=1,inplace = True)


# In[80]:


df[categorical].head()


# In[81]:


df2.head()


# In[82]:


df2 = pd.get_dummies(df2, drop_first=True)


# In[83]:


df2.shape


# In[84]:


df2


# [Table of Contents](#Table-of-Contents)

# # 4. TRAIN - TEST SPLIT & HANDLING WITH MISSING VALUES

# ## 4.1 Train - Test Split
# 
# I must separate the columns (attributes or features) of the dataset into input patterns (X) and output patterns (y).

# In[85]:


X = df2.drop(["Exited"], axis=1)
y = df2["Exited"]


# Finally, I arrived the crucial step in machine learning where the dataset is divided into a training set, used to build models, and a test set, employed to evaluate model performance on new data. This practice, implemented using the train_test_split() function from scikit-learn, ensures that the model is capable of making accurate predictions beyond the training data by simulating real-world scenarios. Notably, specifying a seed for the random number generator is emphasized to maintain reproducibility, enabling consistent results across different executions of the code.

# Train / Test and Split

# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 42)


# ## 4.2 Handling with Missing Values

# In[87]:


missing(df2)


# * In my dataset, there have been no missing values, so there is no need to handle them.

# # 5. FEATURE SCALING

# ## 5.1 The Implementation of Scaling

# In[88]:


X


# In[89]:


y


# # Note.2
# 
# Feature scaling, also known as normalization, is a method employed in data preprocessing to standardize the range of independent variables or features in a dataset. Its primary purpose in machine learning is to ensure that all features are on the same scale, preventing certain algorithms from being sensitive to the magnitude of the features. 
# For example, distance-based algorithms like K-NN and SVM are particularly affected by feature scales, and normalization helps maintain their performance. On the other hand, graphical-model based classifiers (e.g., Fisher LDA, Naive Bayes), decision trees, and tree-based ensemble methods (RF, XGB) are generally insensitive to feature scaling, but it's often recommended to perform scaling for consistency and potential improvement in model performance.
# In essence, feature scaling is a crucial preprocessing step to enhance the robustness and effectiveness of various machine learning algorithms.

# In[90]:


scaler = MinMaxScaler()
scaler


# In[91]:


X_train_scaled = scaler.fit_transform(X_train)


# In[92]:


X_test_scaled = scaler.transform(X_test)


# ## 5.2 General Insights Before Going Further

# In[93]:


# General Insights (training accuracy of each model)

def model_first_insight(X_train, y_train, class_weight, solver='liblinear'):
    # Logistic Regression
    log = LogisticRegression(random_state=42, class_weight=class_weight)
    log.fit(X_train, y_train)
     # SVC
    svc = SVC(random_state=42, class_weight=class_weight)
    svc.fit(X_train, y_train) 
     # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train) 
    # GB GradientBoosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    
    # Model Accuracy on Training Data
    print(f"\033[1m1) Logistic Regression Training Accuracy:\033[0m {log.score(X_train, y_train)}")
    print(f"\033[1m2) SVC Training Accuracy:\033[0m {svc.score(X_train, y_train)}")    
    print(f"\033[1m5) KNN Training Accuracy:\033[0m {knn.score(X_train, y_train)}")
    print(f"\033[1m6) GradiendBoosting Training Accuracy:\033[0m {gb.score(X_train, y_train)}")
      
    return log, svc, knn , gb


# In[94]:


def models(X_train, y_train, class_weight):
    
    # Logistic Regression
    log = LogisticRegression(random_state=42, class_weight=class_weight, solver='liblinear')
    log.fit(X_train, y_train)
    
    # SVC
    svc = SVC(random_state=42, class_weight=class_weight)
    svc.fit(X_train, y_train) 
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train) 
    
     # GB GradientBoosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
     
    # Model Accuracy on Training Data
    print(f"\033[1m1) Logistic Regression Training Accuracy:\033[0m {log}")
    print(f"\033[1m2) SVC Training Accuracy:\033[0m {svc}")    
    print(f"\033[1m5) KNN Training Accuracy:\033[0m {knn}") 
    print(f"\033[1m6) GradiendBoosting Training Accuracy:\033[0m {gb}")
    
    return log.score(X_train, y_train), svc.score(X_train, y_train),knn.score(X_train, y_train), gb.score(X_train, y_train)


# In[95]:


def models_accuracy(X_Set, y_Set):    
    Scores = pd.DataFrame(columns = ["LR_Acc", "SVC_Acc", "KNN_Acc" , "GB_Acc"])

    print("\033[1mBASIC ACCURACY\033[0m")
    Basic = [log_acc, svc_acc, knn_acc, gb_acc] = models(X_train, y_train, None)
    Scores.loc[0] = Basic

    print("\n\033[1mSCALED ACCURACY WITHOUT BALANCED\033[0m")    
    Scaled = [log_acc, svc_acc, knn_acc, gb_acc] = models(X_train_scaled, y_train, None)
    Scores.loc[1] = Scaled

    
    print("\n\033[1mBASIC ACCURACY WITH BALANCED\033[0m")
    Balanced = [log_acc, svc_acc, knn_acc, gb_acc ] = models(X_train, y_train, "balanced")
    Scores.loc[2] = Balanced

    print("\n\033[1mSCALED ACCURACY WITH BALANCED\033[0m")    
    Scaled_Balanced = [log_acc, svc_acc, knn_acc , gb_acc] = models(X_train_scaled, y_train, "balanced")
    Scores.loc[3] = Scaled_Balanced

    Scores.set_axis(['Basic', 'Scaled', 'Balanced', 'Scaled_Balanced'], axis='index', inplace=True)
    #Scores.style.background_gradient(cmap='RdPu')

    return Scores.style.applymap(lambda x: "background-color: pink" if x<0.6 or x == 1 else "background-color: lightgreen")\
                       .applymap(lambda x: 'opacity: 40%;' if (x < 0.8) else None)\
                       .applymap(lambda x: 'color: red' if x == 1 or x <=0.8 else 'color: darkblue')

# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html


# In[96]:


models_accuracy(X_train, y_train)


# In[97]:


Scores = pd.DataFrame(columns = ["LR_Acc", "SVC_Acc", "KNN_Acc", "GB_Acc" ])

print("\033[1mBASIC ACCURACY\033[0m")
Basic = [log_acc, svc_acc, knn_acc , gb_acc ] = models(X_train, y_train, None)
Scores.loc[0] = Basic

print("\n\033[1mSCALED ACCURACY WITHOUT BALANCED\033[0m")    
Scaled = [log_acc, svc_acc, knn_acc , gb_acc ] = models(X_train_scaled, y_train, None)
Scores.loc[1] = Scaled

print("\n\033[1mBASIC ACCURACY WITH BALANCED\033[0m")
Balanced = [log_acc, svc_acc, knn_acc , gb_acc ] = models(X_train, y_train, "balanced")
Scores.loc[2] = Balanced

print("\n\033[1mSCALED ACCURACY WITH BALANCED\033[0m")    
Scaled_Balanced = [log_acc, svc_acc, knn_acc , gb_acc ] = models(X_train_scaled, y_train, "balanced")
Scores.loc[3] = Scaled_Balanced

Scores.set_axis(['Basic', 'Scaled', 'Balanced', 'Scaled_Balanced'], axis='index', inplace=True)


# In[98]:


accuracy_scores = Scores.style.applymap(lambda x: "background-color: pink" if x<0.6 or x == 1 else "background-color: lightgreen")\
                              .applymap(lambda x: 'opacity: 40%;' if (x < 0.8) else None)\
                              .applymap(lambda x: 'color: red' if x == 1 or x <=0.8 else 'color: darkblue')

accuracy_scores


# ## 5.3 Handling with Skewness with PowerTransform & Checking Model Accuracy Scores

# In[99]:


accuracy_scores


# In[100]:


operations = [("scaler", MinMaxScaler()), ("power", PowerTransformer()), ("log", LogisticRegression(random_state=42))]


# In[101]:


# Defining the pipeline object for LogisticClassifier

pipe_log_model = Pipeline(steps=operations)


# In[102]:


pipe_log_model.get_params()


# In[103]:


pipe_log_model.fit(X_train, y_train)
y_pred = pipe_log_model.predict(X_test)
y_train_pred = pipe_log_model.predict(X_train)


# In[104]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **SPECIAL NOTE: When I examine the results after handling with skewness, it's clear to assume that handling with skewness could NOT make any contribution to my model when comparing the results obtained by LogisticClassifier without using PowerTransform. So, for the next steps in this study, I will continue not handling with skewness assuming that it's useless for the results.**

# In[105]:


pipe_scores = cross_validate(pipe_log_model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_pipe_scores = pd.DataFrame(pipe_scores, index = range(1, 11))

df_pipe_scores


# In[106]:


df_pipe_scores.mean()[2:]


# In[107]:


# evaluate the pipeline

# from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, random_state=42)
n_scores = cross_val_score(pipe_log_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print(f'Accuracy: Results Mean : %{round(n_scores.mean()*100,3)}, Results Standard Deviation : {round(n_scores.std()*100,3)}')


# In[108]:


print('Accuracy: %.3f (%.3f)' % (n_scores.mean(), n_scores.std()))


# It's right time to train my models.
# 
# After determining related Classifiers from the scikit-learn framework, I can create and and fit them to my training dataset. Models are fit using the scikit-learn API and the model.fit() function.
# 
# Then I can make predictions using the fit model on the test dataset. To make predictions we use the scikit-learn function model.predict().

# # 6. MODELLING & MODEL PERFORMANCE

# [Table of Contents](#Table-of-Contents)

# ## 6.1 The Implementation of Logistic Regression (LR)

# In[109]:


# take a close look to the models' accuracy scores for comparing the results given by Basic, Scaled, Balanced and Scaled_Balanced models.

accuracy_scores


# ### 6.1.1 Modelling Logistic Regression (LR) with Default Parameters

# In[110]:


from sklearn.metrics import ConfusionMatrixDisplay,roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score

LR_model = LogisticRegression() # Since Scaled accuracy outcome gives the best model accuracy results, I will implement it 
LR_model.fit(X_train_scaled, y_train)
y_pred = LR_model.predict(X_test_scaled)
y_train_pred = LR_model.predict(X_train_scaled)

log_f1 = f1_score(y_test, y_pred)
log_acc = accuracy_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(LR_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# In[111]:


# analyzing class probabilities

y_pred_proba = LR_model.predict_proba(X_test_scaled)
y_pred_proba


# In[112]:


test_data = pd.concat([X_test.set_index(y_test.index), y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba"] = y_pred_proba[:, 1]
test_data.sample(10)


# ### 6.1.2 Cross-Validating Logistic Regression (LR) Model

# In[113]:


# examination of model scores across multiple performance metrics and folds

log_xvalid_model = LogisticRegression()

log_xvalid_model_scores = cross_validate(log_xvalid_model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                          'f1'], cv = 10)
log_xvalid_model_scores = pd.DataFrame(log_xvalid_model_scores, index = range(1, 11))

log_xvalid_model_scores


# In[114]:


log_xvalid_model_scores.mean()[2:]


# ### 6.1.3 Modelling Logistic Regression (LR) with Best Parameters Using GridSeachCV

# In[115]:


penalty = ["l1", "l2", "elasticnet"]
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0, 10, 20)

param_grid = {"penalty" : penalty,
             "l1_ratio" : l1_ratio,
             "C" : C}


# In[116]:


LR_grid_model = LogisticRegression(solver='saga', max_iter=5000, class_weight = "balanced")

LR_grid_model = GridSearchCV(LR_grid_model, param_grid = param_grid)


# In[117]:


LR_grid_model.fit(X_train_scaled, y_train)


# Let's look at the best parameters & estimator found by GridSearchCV.

# In[118]:


print(colored('\033[1mBest Parameters of GridSearchCV for LR Model:\033[0m', 'blue'), colored(LR_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for LR Model:\033[0m', 'blue'), colored(LR_grid_model.best_estimator_, 'cyan'))


# In[119]:


y_pred = LR_grid_model.predict(X_test_scaled)
y_train_pred = LR_grid_model.predict(X_train_scaled)

log_grid_f1 = f1_score(y_test, y_pred)
log_grid_acc = accuracy_score(y_test, y_pred)
log_grid_recall = recall_score(y_test, y_pred)
log_grid_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(LR_grid_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# ### 6.1.4 ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[181]:


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(LR_grid_model, X_test_scaled, y_test, response_method='auto');

# In machine learning, AUC stands for Area Under the Receiver Operating Characteristic Curve.
# It is a metric used to evaluate the performance of a binary classification model. The AUC represents the area under the curve when plotting the Receiver Operating Characteristic (ROC) curve,
# which illustrates the trade-off between the true positive rate (sensitivity) and the false positive rate at various classification thresholds.

# An AUC value of 0.79 indicates the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
# It suggests that the model has good discriminatory power. It indicates that, on average, the model is effective at distinguishing between positive and negative instances. 


# In[182]:


from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(LR_grid_model, X_test_scaled, y_test);

# In machine learning, AP stands for Average Precision. 
# It is a metric used to evaluate the performance of a binary classification model, particularly when dealing with imbalanced datasets.
# The Average Precision is calculated based on the precision-recall curve.

# An AP value of 0.50 indicates the average precision across different recall levels. 
# The AP value ranges from 0 to 1, where a higher AP implies better model performance. 
# In my case, an AP of 0.50 suggests that, the model's ability to correctly classify positive instances is moderate but might still need improvement. 


# ### 6.1.5 The Determination of The Optimal Threshold

# In[122]:


fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred_proba[:, 1])


# In[123]:


optimal_idx = np.argmax(tp_rate - fp_rate)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


# In[124]:


roc_curve = {"fp_rate":fp_rate, "tp_rate":tp_rate, "thresholds":thresholds}
df_roc_curve = pd.DataFrame(roc_curve)
df_roc_curve


# In[125]:


df_roc_curve.iloc[optimal_idx]


# [Table of Contents](#Table-of-Contents)

# ## 6.2 The Implementation of Support Vector Machine (SVM)

# In[126]:


# take a close look to the models' accuracy scores for comparing the results given by Basic, Scaled, Balanced and Scaled_Balanced models.

accuracy_scores


# ### 6.2.1 Modelling Support Vector Machine (SVM) with Default Parameters

# In[127]:


SVM_model = SVC(random_state=42)
SVM_model.fit(X_train_scaled, y_train)
y_pred = SVM_model.predict(X_test_scaled)
y_train_pred = SVM_model.predict(X_train_scaled)

svm_f1 = f1_score(y_test, y_pred)
svm_acc = accuracy_score(y_test, y_pred)
svm_recall = recall_score(y_test, y_pred)
svm_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(SVM_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# In[128]:


# cross-checking the model by predictions in Train Set for consistency

y_train_pred = SVM_model.predict(X_train_scaled)

print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(SVM_model, X_train_scaled, y_train);


# In[129]:


get_ipython().system('pip install yellowbrick')
from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(SVM_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# ### 6.2.2 Cross-Validating Support Vector Machine (SVM) Model

# In[130]:


# examination of model scores across multiple performance metrics and folds

svm_xvalid_model = SVC()

svm_xvalid_model_scores = cross_validate(svm_xvalid_model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                   'f1'], cv = 10)
svm_xvalid_model_scores = pd.DataFrame(svm_xvalid_model_scores, index = range(1, 11))

svm_xvalid_model_scores


# In[131]:


svm_xvalid_model_scores.mean()[2:]


# ### 6.2.3 Modelling Support Vector Machine (SVM) with Best Parameters Using GridSeachCV

# In[132]:


# hyperparameter tuning is crucial for achieving the best possible model performance in machine learning tasks.

param_grid = {'C': [0.1,1, 10, 100, 1000],
              'gamma': ["scale", "auto", 1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf', 'linear']}


# In[133]:


SVM_grid_model = SVC(random_state=42)

SVM_grid_model = GridSearchCV(SVM_grid_model, param_grid, verbose=3, refit=True)


# In[134]:


# it took 5 hours to complete

SVM_grid_model.fit(X_train_scaled, y_train)


# Let's look at the best parameters & estimator found by GridSearchCV.

# In[135]:


print(colored('\033[1mBest Parameters of GridSearchCV for SVM Model:\033[0m', 'blue'), colored(SVM_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for SVM Model:\033[0m', 'blue'), colored(SVM_grid_model.best_estimator_, 'cyan'))


# In[136]:


y_pred = SVM_grid_model.predict(X_test_scaled)
y_train_pred = SVM_grid_model.predict(X_train_scaled)

svm_grid_f1 = f1_score(y_test, y_pred)
svm_grid_acc = accuracy_score(y_test, y_pred)
svm_grid_recall = recall_score(y_test, y_pred)
svm_grid_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(SVM_grid_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# **GridSearchCV contributed to an improvement in True Positive predictions, increasing from 174 to 239, while False Negative predictions decreased from 437 to 372.**

# ### 6.2.4 ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[137]:


RocCurveDisplay.from_estimator(SVM_grid_model, X_test_scaled, y_test);

# An AUC of 0.84 indicates that, on average, the model is effective at distinguishing between positive and negative instances. 
# It suggests a reasonably good performance


# In[138]:


PrecisionRecallDisplay.from_estimator(SVM_grid_model, X_test_scaled, y_test);

# An AP of 0.68 suggests that, on average, the model's precision is acceptable.
# It indicates that the model is reasonably effective at correctly classifying positive instances.


# [Table of Contents](#Table-of-Contents)

# ## 6.3 The Implementation of K-Nearest Neighbor (KNN)

# In[139]:


# take a close look to the models' accuracy scores for comparing the results given by Basic, Scaled, Balanced and Scaled_Balanced models.

accuracy_scores


# ### 6.3.1 Modelling K-Nearest Neighbor (KNN) with Default Parameters

# In[140]:


KNN_model = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
KNN_model.fit(X_train_scaled, y_train)
y_pred = KNN_model.predict(X_test_scaled)
y_train_pred = KNN_model.predict(X_train_scaled)

knn_f1 = f1_score(y_test, y_pred)
knn_acc = accuracy_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(KNN_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# In[141]:


y_pred_proba = KNN_model.predict_proba(X_test_scaled)


# In[142]:


pd.DataFrame(y_pred_proba)


# In[143]:


my_dict = {"Actual": y_test, "Pred": y_pred, "Proba_1": y_pred_proba[:,1], "Proba_0":y_pred_proba[:,0]}


# In[144]:


pd.DataFrame.from_dict(my_dict).sample(10)


# ### 6.3.2 Cross-Validating K-Nearest Neighbor (KNN)

# In[145]:


knn_xvalid_model = KNeighborsClassifier(n_neighbors=5)

knn_xvalid_model_scores = cross_validate(knn_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
knn_xvalid_model_scores = pd.DataFrame(knn_xvalid_model_scores, index = range(1, 11))

knn_xvalid_model_scores


# In[146]:


knn_xvalid_model_scores.mean()[2:]


# ### 6.3.3 Elbow Method for Choosing Reasonable K Values

# In[147]:


test_error_rates = []


for k in range(1, 30):
    KNN_model = KNeighborsClassifier(n_neighbors=k)
    KNN_model.fit(X_train_scaled, y_train) 
   
    y_test_pred = KNN_model.predict(X_test_scaled)
    
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_error_rates.append(test_error)


# In[148]:


test_error_rates


# In[149]:


plt.figure(figsize=(15, 8))
plt.plot(range(1, 30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.18733333333333335, xmin=0, xmax=30, colors='r', linestyles="--")
plt.hlines(y=0.18000000000000005, xmin=0, xmax=30, colors='r', linestyles="--");


# ### 6.3.4 GridsearchCV for Choosing Reasonable K Values

# In[150]:


k_values= range(1, 30)
param_grid = {"n_neighbors": k_values, "p": [1, 2], "weights": ['uniform', "distance"]}


# In[151]:


KNN_grid = KNeighborsClassifier()


# In[152]:


KNN_grid_model = GridSearchCV(KNN_grid, param_grid, cv=10, scoring='accuracy')


# In[153]:


KNN_grid_model.fit(X_train_scaled, y_train)


# Let's look at the best parameters & estimator found by GridSearchCV.

# In[154]:


print(colored('\033[1mBest Parameters of GridSearchCV for KNN Model:\033[0m', 'blue'), colored(KNN_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for KNN Model:\033[0m', 'blue'), colored(KNN_grid_model.best_estimator_, 'cyan'))


# In[155]:


# NOW WITH K=8

KNN_model = KNeighborsClassifier(n_neighbors=8, p=1)
KNN_model.fit(X_train_scaled, y_train)
pred = KNN_model.predict(X_test_scaled)
y_train_pred = KNN_model.predict(X_train_scaled)

knn8_f1 = f1_score(y_test, y_pred)
knn8_acc = accuracy_score(y_test, y_pred)
knn8_recall = recall_score(y_test, y_pred)
knn8_auc = roc_auc_score(y_test, y_pred)

print('WITH K=8')
print('-------------------')
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(KNN_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# ### 6.3.5 ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[156]:


RocCurveDisplay.from_estimator(KNN_model, X_test_scaled, y_test);

# An AUC of 0.77 suggests that, on average, the model is relatively effective at distinguishing between positive and negative instances. 
# While it may not be considered excellent, it indicates a reasonable level of performance.


# In[157]:


PrecisionRecallDisplay.from_estimator(KNN_model, X_test_scaled, y_test);

# An AP of 0.47 indicates that, on average, the model's precision is relatively low across different recall levels. 
# It suggests that the model may struggle to precisely identify positive instances, and there is room for improvement. 


# [Table of Contents](#Table-of-Contents)

# ## 6.4 The Implementation of GradientBoosting (GB)

# In[158]:


# take a close look to the models' accuracy scores for comparing the results given by Basic, Scaled, Balanced and Scaled_Balanced models.

accuracy_scores


# ### 6.4.1 Modelling GradientBoosting (GB) with Default Parameters

# In[159]:


GB_model = GradientBoostingClassifier(random_state=42)
GB_model.fit(X_train_scaled, y_train)
y_pred = GB_model.predict(X_test_scaled)
y_train_pred = GB_model.predict(X_train_scaled)

gb_f1 = f1_score(y_test, y_pred)
gb_acc = accuracy_score(y_test, y_pred)
gb_recall = recall_score(y_test, y_pred)
gb_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(GB_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# Cross-checking the model by predictions in Train Set for consistency

# In[160]:


y_train_pred = GB_model.predict(X_train_scaled)

print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(GB_model, X_train_scaled, y_train);


# In[161]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(GB_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# ### 6.4.2 Cross-Validating GradientBoosting (GB)

# In[162]:


gb_xvalid_model = GradientBoostingClassifier(random_state=42)

gb_xvalid_model_scores = cross_validate(gb_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"], cv = 10)
gb_xvalid_model_scores = pd.DataFrame(gb_xvalid_model_scores, index = range(1, 11))

gb_xvalid_model_scores


# In[163]:


gb_xvalid_model_scores.mean()


# ### 6.4.3 Feature Importance for GradientBoosting (GB) Model

# In[164]:


# the importance scores of each feature

GB_model.feature_importances_


# In[165]:


GB_feature_imp = pd.DataFrame(index = X.columns, data = GB_model.feature_importances_,
                              columns = ["Feature Importance"]).sort_values("Feature Importance", ascending = False)
GB_feature_imp


# In[166]:


sns.barplot(y=GB_feature_imp["Feature Importance"], x=GB_feature_imp.index)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# ### 6.4.4 Modelling GradientBoosting (GB) Model with Best Parameters Using GridSeachCV

# In[167]:


# Computing the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.75, 0.85, 1]

for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, random_state=42)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, y_test)))
    print()


# In[168]:


param_grid = {"n_estimators":[100, 200, 300],
             "subsample":[0.5, 1], "max_features" : [None, 2, 3, 4], "learning_rate": [0.2, 0.5, 0.6, 0.75, 0.85, 1.0, 1.25, 1.5]}  # 'max_depth':[3,4,5,6]


# In[169]:


GB_grid_model = GradientBoostingClassifier(random_state=42)

GB_grid_model = GridSearchCV(GB_grid_model, param_grid, scoring = "f1", verbose=2, n_jobs = -1).fit(X_train, y_train)


# Let's look at the best parameters & estimator found by GridSearchCV.

# In[170]:


print(colored('\033[1mBest Parameters of GridSearchCV for Gradient Boosting Model:\033[0m', 'blue'), colored(GB_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for Gradient Boosting Model:\033[0m', 'blue'), colored(GB_grid_model.best_estimator_, 'cyan'))


# In[180]:


y_pred = GB_grid_model.predict(X_test_scaled)
y_train_pred = GB_grid_model.predict(X_train_scaled)

gb_grid_f1 = f1_score(y_test, y_pred)
gb_grid_acc = accuracy_score(y_test, y_pred)
gb_grid_recall = recall_score(y_test, y_pred)
gb_grid_auc = roc_auc_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

ConfusionMatrixDisplay.from_estimator(GB_grid_model, X_test_scaled, y_test)

train_val(y_train, y_train_pred, y_test, y_pred)


# # Note.3
# 
# The confusion matrix shows that the model is predicting all instances as class 1, resulting in 0 true negatives (TN), 2389 false positives (FP), 0 false negatives (FN), and 611 true positives (TP). This leads to precision, recall, and F1-score values that are not representative of a well-performing model. The reason behind this issue is Imbalanced Classes; the dataset seems highly imbalanced, with class 0 having a significantly larger number of instances than class 1. The model is predicting only class 1, leading to a high recall for class 1 but very low precision and recall for class 0. Evaluation Metrics such as precision, recall, and F1-score for class 0 are severely impacted due to the lack of true negatives in the predictions. For addressing Imbalanced Classes, I will use different evaluation metrics; area under the Precision-Recall curve (AUC-PR).
# The precision for class 0 is not calculable due to the absence of true positives for this class. The metrics suggest that the hyperparameter tuning performed using GridSearchCV may not have effectively addressed the issue of class imbalance or model behavior.

# ### 6.4.5 ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[172]:


RocCurveDisplay.from_estimator(GB_model, X_test, y_test);

#  An AUC of 0.36 suggests that, on average, the model struggles to distinguish between positive and negative instances.


# In[173]:


PrecisionRecallDisplay.from_estimator(GB_model, X_test, y_test);

# An AP of 0.16 indicates that, on average, the model's precision is relatively low across different recall levels.


# [Table of Contents](#Table-of-Contents)

# # 7. THE COMPARISON OF MODELS

# In[188]:


compare = pd.DataFrame({"Model": ["Logistic Regression", "SVM", "KNN" , "GradientBoost"],
                        "F1": [log_grid_f1, svm_grid_f1, knn_f1 , gb_f1 ],
                        "Recall": [log_grid_recall, svm_grid_recall, knn_recall, gb_recall ],
                        "Accuracy": [log_grid_acc, svm_grid_acc, knn_acc, gb_acc ],
                        "ROC_AUC": [log_grid_auc, svm_grid_auc, knn_auc, gb_auc ]})

def labels(ax):
    for p in ax.patches:
        width = p.get_width()                        # get bar length
        ax.text(width,                               # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                '{:1.3f}'.format(width),             # set variable to display, 2 decimals
                ha = 'left',                         # horizontal alignment
                va = 'center')                       # vertical alignment
    
plt.figure(figsize=(14,14))
plt.subplot(411)
compare = compare.sort_values(by="F1", ascending=False)
ax=sns.barplot(x="F1", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(412)
compare = compare.sort_values(by="Recall", ascending=False)
ax=sns.barplot(x="Recall", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(413)
compare = compare.sort_values(by="Accuracy", ascending=False)
ax=sns.barplot(x="Accuracy", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(414)
compare = compare.sort_values(by="ROC_AUC", ascending=False)
ax=sns.barplot(x="ROC_AUC", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.show()


# * The F1 score is a metric that combines precision and recall into a single value, providing a balance between the two. In general:
# 
# _A higher F1 score indicates a better balance between precision and recall._
# _A lower F1 score suggests an imbalance or trade-off between precision and recall._
# 
# ***The F1 score of the GradientBoosting model (0.614) is relatively high, indicating a good balance between precision and recall. The SVM's F1 score (0.528) is moderately high and also suggests a decent balance between precision and recall. Logistic Regression's F1 score (0.513) suggests that the model has a reasonable balance between precision and recall, In contrast, the F1 score of KNN model is below 0.5, which is relatively lower than the scores of the previous three models, indicating a weaker balance between precision and recall.***
# 
# * Recall, also known as sensitivity or true positive rate, measures the ability of a model to capture all positive instances. In general:
# 
# _A higher recall score indicates a better ability to identify positive instances._
# _A lower recall score suggests that the model is missing a significant portion of positive instances._
# 
# ***Logistic Regression's recall score (0.728) is the highest among the provided scores, suggesting the strongest ability to identify positive instances. The recall score of the GradientBoosting model (0.498) is relatively high, indicating a good ability to capture positive instances. The SVM model's recall score (0.391) is moderate, suggesting a reasonable ability to identify positive instances. The recall score for the KNN model (0.316) is lower than those of the previous three, indicating a weaker ability to capture positive instances.***
# 
# * Accuracy is a measure of the overall correctness of the model, representing the ratio of correctly predicted instances to the total instances. In general:
# 
# _A higher accuracy score indicates better overall performance._
# _A lower accuracy score suggests that the model is making more mistakes._
# 
# ***The accuracy score of the GradientBoosting model (0.872) is relatively high, indicating strong overall performance. The accuracy score of the SVM model (0.858) is also high and suggests a good overall performance, though slightly lower than that of the GradientBoosting model. The accuracy score of the KNN model (0.815), it indicates that the model made correct predictions for approximately 81.5% of the instances. And finally the accuracy score of Linear Regression (0.719), is the lowest comparing the previous three models, indicating a decrease in overall performance.***
# 
# * The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score measures the ability of a model to distinguish between positive and negative classes. In general:
# 
# _A higher ROC AUC score indicates better discriminatory power of the model._
# _A lower ROC AUC score suggests weaker discrimination._
# 
# ***The ROC AUC score of the GradientBoosting model (0.733) is relatively high, indicating good discriminatory power. The ROC AUC score of the Linear Regression model (0.722) is slightly lower than that of the GradientBoosting model, suggesting somewhat weaker discriminatory power. The ROC AUC scores of the remaining models, SVM model (0.684) and KNN (0.629), are lower than those of the previous two, indicating a decrease in discriminatory power.***
# 
# > **These results reveal distinct performance characteristics among different models. The GradientBoosting model emerges as a top performer with a high F1 score (0.614), indicating a commendable balance between precision and recall, a relatively high recall score (0.498), and strong accuracy (0.872) and ROC AUC (0.733) scores, demonstrating robust overall performance and discriminatory power. 
# The SVM model follows closely with a moderately high F1 score (0.528), good recall (0.391), high accuracy (0.858), and a respectable ROC AUC score (0.684). Logistic Regression showcases a reasonable balance between precision and recall (F1 score: 0.513) and the highest recall score (0.728), emphasizing its strength in identifying positive instances. 
# In contrast, the KNN model exhibits a weaker balance between precision and recall (F1 score: below 0.5) and lower recall (0.316), while Linear Regression demonstrates the lowest accuracy (0.719) among the considered models.**

# # 8. CONCLUSION

# In my study ,
# 
# > I have tried to a predict classification problem in Bank Customer Churn Dataset by a variety of models to classify Churn predictions in the context of determining whether a customer is likely to be churned based on the input parameters like gender, balance and various test results or not.
# 
# > I have made the detailed exploratory analysis (EDA).
# 
# > There have been NO missing values in the Dataset.
# 
# > I have decided which metrics will be used.
# 
# > I have analyzed both target and features in detail.
# 
# > I have transformed categorical variables into dummies so we can use them in the models.
# 
# > I have handled with skewness problem for make them closer to normal distribution; however, having examined the results, it's clear to assume that handling with skewness could NOT make any contribution to our models when comparing the results obtained by LogisticClassifier without using PowerTransform. Therefore, in this study I have continue not handling with skewness assuming that it's useless for the results.
# 
# > I have cross-checked the models obtained from train sets by applying cross validation for each model performance.
# 
# > I have examined the feature importance of some models.
# 
# > Lastly I have examined the results of all models visually with respect to select the best one for my problem.

# In[175]:


#Execution started:19:00--14:17
#Execution ended:09:25--19:00


# In[176]:


#Two last line of Notebook
end = time.time()
elapsed_time(start, end)


# # 9. REFERENCES
# * https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction?resource=download 
# * https://www.kaggle.com/code/azminetoushikwasi/classification-comparing-different-algorithms
# * https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques
# * https://www.codeavail.com/blog/data-science-project-ideas/
# * https://www.analytixlabs.co.in/blog/data-science-projects/#project3
# * https://www.kaggle.com/code/leemun1/predicting-breast-cancer-logistic-regression/notebook
# * https://www.kaggle.com/code/mouadberqia/bank-churn-prediction-beginner-friendly-0-88959

# [Table of Contents](#Table-of-Contents)

# In[ ]:




