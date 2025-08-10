
#importing the necessary libraries

import pandas as pd #used to store data
import numpy as np #used to do array calculations
import matplotlib.pyplot as plt #used for visualizing
import seaborn as sb #also used fr visualization

from sklearn.model_selection import train_test_split #used to split training data and testing data
from sklearn.preprocessing import LabelEncoder, StandardScaler #LabelEncoder is used to encode categorical labels to int, StandardScaler sets the mean=0 and s.d = 1 for all variables, ensuirng theyre on the sames scale for optimal operation of the model
from sklearn import metrics #metrics contains many evaluation function used to evaluate the model's perfomance
from sklearn.svm import SVC #support vector classifier, used to classify
from sklearn.metrics import mean_absolute_error as mae #lower mean absolute error equals better model perfomance
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor #Ensemble is a ML technique that combines multiple ml models to produce one stronger, more accurate model

import warnings
warnings.filterwarnings('ignore') #filters out warnings that come with the word "ignore"

from datetime import datetime


# In[130]:


#reading the ola data

df = pd.read_csv(r'C:\Users\Jackson Jesse\Desktop\everything\Sem 5\MLOps\ML\Ola ride request forecasting\ola.csv') #use raw string 'r' while entering file paths
df.head()


# In[131]:


#Feature Engineering - used to derive multiple features from existing features

parts = df['datetime'].str.split(" ", n=2, expand=True)
df['date'] = parts[0]
df['time'] = parts[1].str[:2].astype('int')

date_parts = df['date'].str.split("-",n=3, expand=True)
df['year'] = date_parts[0].astype('int')
df['month'] = date_parts[1].astype('int')
df['date'] = date_parts[2].astype('int')


# In[132]:


#Since the rides frequency can differ based on its a weekday or a weeknd, we are segregating them

df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

def weekend_or_weekday(year, month, day):
    try:
        d = datetime(year, month, day)
        if d.weekday() > 4: #weekday return the index of the day of the weel, mon - 0, sun -6
            return 0
        else:
            return 1
    except ValueError:
        return np.nan

df['weekday'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis = 1)


# In[133]:


df.tail()


# In[134]:


#checking the time
def am_or_pm(x):
    if x > 11:
        return 1
    else:
        return 0
df['am_or_pm'] = df['time'].apply(am_or_pm)
df.head()


# In[135]:


get_ipython().system('pip install holidays')


# In[136]:


import holidays
india_holidays = holidays.country_holidays("IN")
def is_holiday(x):
    if india_holidays.get(x.date()):
        return 1
    else:
        return 0
df['holidays'] = df['datetime'].apply(is_holiday)


# In[137]:


df[df['holidays'] == 1]


# In[138]:


df.drop('date', axis=1, inplace=True)


# # Step 3
# Exploratory Data Analysis - EDA
# analyzing data visually

# In[140]:


df.isna().sum()


# In[141]:


#checking the relation beetween ride request count and the day, time or month

features = ['day', 'time', 'month']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(3, 1, i+1)
    df.groupby(col).mean()['count'].plot()
plt.show()


# In[142]:


features = ['season', 'weather', 'holidays','am_or_pm', 'year', 'weekday']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(3, 2, i+1)
    df.groupby(col).mean()['count'].plot.bar()
plt.show()


# In[143]:


features = ['temp', 'windspeed']
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 2, i+1)
    sb.distplot(df[col])
plt.show()


# In[144]:


# number of rows that will be lost if we remove the utliers
df.shape[0] - df[df['windspeed']<32].shape[0]


# In[145]:


features = ['humidity', 'casual', 'registered', 'count']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    sb.boxplot(data=df[col])
plt.show()


# In[146]:


sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()


# In[147]:


df.drop(['registered', 'time'], axis=1, inplace=True)


# In[148]:


df = df[(df['windspeed'] < 32) & (df['humidity'] > 0)]


# # Model training

# In[ ]:


df.drop('datetime', axis=1, inplace=True)


# In[156]:


features = df.drop(['count'], axis=1)
target = df['count'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.2, random_state=47)
X_train.shape, Y_train.shape


# In[158]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[154]:





# In[98]:





# In[124]:


print(df.dtypes)


# In[162]:


#Models evaluating
models = [LinearRegression(), Lasso(), RandomForestRegressor(), Ridge()]
val_error = []
for i in models:
    i.fit(X_train, Y_train)
    print(f'{i}: ')
    train_preds = i.predict(X_train)
    print('Training Error: ', mae(Y_train, train_preds))

    val_preds = i.predict(X_val)
    print('Validation error: ', mae(Y_val, val_preds))
    val_error.append(mae(Y_val, val_preds))
    print()


# In[176]:


#finding the best model
print(f"The model to choose is the one with the lowest mae, which is '{models[val_error.index(min(val_error))]}'")


# In[187]:


#choosing th best model
# best_model = RandomForestRegressor()
best_model = Ridge()
best_model.fit(X_train, Y_train)


# In[189]:


#predicting
y_pred = best_model.predict(X_val)


# In[191]:


print(y_pred)


# In[193]:


for actual, predicted in zip(Y_val[:10], y_pred[:10]):
    print(f'Actual: {actual:.2f}, Predicted: {predicted:.2f}')


# In[195]:


import joblib
joblib.dump(best_model, 'firstModel.pkl')


# In[ ]:




