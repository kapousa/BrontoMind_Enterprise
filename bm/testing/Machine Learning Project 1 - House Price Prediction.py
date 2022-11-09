# Machine Learning Project 1 - House Price Prediction

import pandas as pd

df1 = pd.read_csv('bengaluru_house_prices.csv')

df1.head()

df1.info()

df1.shape

df2 = df1.drop(['area_type', 'society', 'balcony'], axis=1)

df2.head()

df2.isnull().sum()

df3 = df2.dropna()

df3.isnull().sum()

df3.head()

df3['availability'].unique()

df3.groupby("availability")["availability"].count()

df3['availability'] = df3['availability'].apply(lambda x: x if x in ('Ready To Move') else 'Future Possession')

df3.groupby("availability")["availability"].count()

df3['location'].unique()

df3.groupby("location")["location"].count().sort_values(ascending=False)

locations = df3.groupby('location')['location'].count().sort_values()

locations

locations_20cnt = locations[locations <= 20]

locations_20cnt

df3['location'] = df3['location'].apply(lambda x: 'Others' if x in locations_20cnt else x)

df3.groupby("location")["location"].count().sort_values(ascending=False)

df3.head()

df3['size'].unique()

import re

df3['bhks'] = df3['size'].apply(lambda x: int(re.findall('\d+', x)[0].strip()))

df3.head()

df3['total_sqft'].unique()


def get_mean(x):
    if re.findall('-', x):
        ss = x.strip().split('-')
        return ((float(ss[0]) + float(ss[0])) / 2)
    try:
        return float(x.strip())
    except:
        return None


df3['total_sqft_new'] = df3['total_sqft'].apply(get_mean)

df3.head()

df3.isnull().sum()

df4 = df3.dropna()

df4.isnull().sum()

df4['bath'].unique()

df4.groupby('bath')['bath'].count().sort_values()

df5 = df4[df4['bath'] <= 10]

df5.head()

df6 = df5.drop(['size', 'total_sqft'], axis=1)

df6.head()

df6[df6['total_sqft_new'] / df6['bhks'] < 400]

df7 = df6[df6['total_sqft_new'] / df6['bhks'] > 400]

df7.head()

df7['price_per_sqft'] = df7['price'] * 100000 / df7['total_sqft_new']

df7

df7['price_per_sqft'].describe()


def rmv_price_outlier(df):
    df_new = pd.DataFrame()
    for key, sdf in df.groupby('location'):
        m = sdf['price_per_sqft'].mean()
        s = sdf['price_per_sqft'].std()
        # print (sdf['location'])
        rdf = sdf[(sdf['price_per_sqft'] <= m + s) & (sdf['price_per_sqft'] > m - s)]
        # print(rdf)
        df_new = pd.concat([df_new, rdf], ignore_index=True)
    return df_new


df8 = rmv_price_outlier(df7)

df8.head()

df8.shape

availability_dummy = pd.get_dummies(df8['availability'], drop_first=True)

availability_dummy

location_dummy = pd.get_dummies(df8['location'], drop_first=True)

df9 = pd.concat([df8, availability_dummy, location_dummy], axis=1)

df9.head()

df10 = df9.drop(['availability', 'location', 'price_per_sqft'], axis=1)

df10.head()

X = df10.drop(['price'], axis=1)
y = df10['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

len(X_train)

len(X_test)

X_train.describe()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

model_score = cross_val_score(estimator=LinearRegression(), X=X_train, y=y_train, cv=6)
model_score

model_score.mean()

model_score.std()

model_score = cross_val_score(estimator=DecisionTreeRegressor(), X=X_train, y=y_train, cv=5)
model_score

model_score.mean()

model_score.std()

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
