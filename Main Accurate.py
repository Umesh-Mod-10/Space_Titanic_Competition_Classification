# %% Importing all the necessary libraries:

from itertools import product
import scipy.stats as ss
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import xgboost as xg
from sklearn.ensemble import RandomForestClassifier

# %% Loading the dataset:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/Space Titanic/train.csv")
test = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Space Titanic\test.csv")

# %% Getting the basic info of the dataset:

print(data.info())
stats = (data.describe())
print(data.shape)
print(data.duplicated().sum())
print(data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False))

# %% Calculating the Nan values Percentages:

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Creating new Useful columns:

data['Cabin'] = data['Cabin'].replace(np.nan, "Z/Z/Z")
data['Cabin'] = data['Cabin'].str.split('/')
data['Cabin_Side'] = data['Cabin'].apply(lambda x: x[-1])
data['Cabin_Side'] = data['Cabin_Side'].replace("Z", np.nan)
data['Cabin_Num'] = data['Cabin'].apply(lambda x: x[-2])
data['Cabin_Num'] = data['Cabin_Num'].replace("Z", np.nan)
data['Cabin_Deck'] = data['Cabin'].apply(lambda x: x[0])
data['Cabin_Deck'] = data['Cabin_Deck'].replace("Z", np.nan)

# %% Getting the Details about the derived columns:

print(data['HomePlanet'].value_counts())

print(data['Cabin_Deck'].value_counts())
t = data[data['Cabin_Deck'] == 'G']

# %% Dropping unnecessary Data columns:

print(data[['Cabin_Side', 'Cabin_Deck', 'Cabin_Num']].nunique())
data.drop(['Name', 'Cabin_Num', 'Cabin', 'PassengerId'], axis=1, inplace=True)

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with the cost Nan values with CryoSleep:

Expences = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Expences:
    mask = data[i].isna() & data['CryoSleep']
    data.loc[mask, i] = 0

df = data.select_dtypes(exclude='object')
nan = df.isna().sum()[df.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with other Cost Nan values:

imputer_median = SimpleImputer(strategy='median')
for i in Expences:
    data[i] = imputer_median.fit_transform(data[[i]])

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Getting a new column of Total Cost:

data['Total_Spend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']

# %% Let's Check the Relation with Spending and CryoSleep:

rel_cryo = data[data['CryoSleep'].isna()]

'''
We found out that if Total_Spending != 0, that person isn't sleeping.
'''

data['CryoSleep'] = data.apply(lambda x: False if x['Total_Spend'] != 0 else True, axis=1)

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with Age Nan with median values:

data['Age'] = imputer_median.fit_transform(data[['Age']])

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %%

col1 = ['HomePlanet', 'Cabin_Side', 'Cabin_Deck', 'Destination', 'Transported', 'VIP']
col2 = col1.copy()

final = list(product(col1, col2, repeat=1))

corr_categorical = []
for var1, var2 in final:
    if var1 != var2:
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        corr_categorical.append((var1, var2, p))

corr_categorical_df = pd.DataFrame(corr_categorical, columns=['var1', 'var2', 'coeff'])
corr_categorical = corr_categorical_df.pivot(index='var1', columns='var2', values='coeff')
corr_categorical_df = corr_categorical_df.sort_values(by='coeff')

'''
By the finding the correlation, we get that VIP and Destination both are highly dependent
on the column of Cabin_Side. Thus we use this for dealing with imputation of Nan values.
'''

# %% Dealing with the Nan values with column of Cabin:

mode = lambda x: x.mode().iloc[0] if not x.mode().empty else None

data.loc[:, 'Cabin_Side'] = data.loc[:, 'Cabin_Side'].fillna(data.groupby('Destination')['Cabin_Side'].transform(mode))
data.loc[:, 'Cabin_Deck'] = data.loc[:, 'Cabin_Deck'].fillna(data.groupby('Destination')['Cabin_Deck'].transform(mode))

data.loc[:, 'Cabin_Side'] = data.loc[:, 'Cabin_Side'].fillna(data.groupby('VIP')['Cabin_Side'].transform(mode))
data.loc[:, 'Cabin_Deck'] = data.loc[:, 'Cabin_Deck'].fillna(data.groupby('VIP')['Cabin_Deck'].transform(mode))

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with the Nan values of the column of VIP:

mode = lambda x: x.mode().iloc[0] if not x.mode().empty else None

data.loc[:, 'VIP'] = data.loc[:, 'VIP'].fillna(data.groupby('Cabin_Side')['VIP'].transform(mode))

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with Nan values of Destination:

data.loc[:, 'Destination'] = data.loc[:, 'Destination'].fillna(data.groupby('Cabin_Side')['Destination'].transform(mode))

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with Nan values:

data.loc[:, 'HomePlanet'] = data.loc[:, 'HomePlanet'].fillna(data.groupby('Cabin_Side')['HomePlanet'].transform(mode))

nan = data.isna().sum()[data.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Getting the test dataset:

print(test.info())
stats_test = (test.describe())
print(test.shape)
print(test.duplicated().sum())
print(test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False))

# %% Calculating the Nan values Percentages:

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Creating new Useful columns:

test['Cabin'] = test['Cabin'].replace(np.nan, "Z/Z/Z")
test['Cabin'] = test['Cabin'].str.split('/')
test['Cabin_Side'] = test['Cabin'].apply(lambda x: x[-1])
test['Cabin_Side'] = test['Cabin_Side'].replace("Z", np.nan)
test['Cabin_Num'] = test['Cabin'].apply(lambda x: x[-2])
test['Cabin_Num'] = test['Cabin_Num'].replace("Z", np.nan)
test['Cabin_Deck'] = test['Cabin'].apply(lambda x: x[0])
test['Cabin_Deck'] = test['Cabin_Deck'].replace("Z", np.nan)

# %% Dropping unnecessary Data columns:

Y_prediction = test['PassengerId']
print(test[['Cabin_Side', 'Cabin_Deck', 'Cabin_Num']].nunique())
test.drop(['Name', 'Cabin_Num', 'Cabin', 'PassengerId'], axis=1, inplace=True)

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %%

test_cat = test.select_dtypes(include='object')
test_num = test.select_dtypes(exclude='object')

# %% Dealing with the cost Nan values with CryoSleep:

Expences = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Expences:
    mask = test[i].isna() & data['CryoSleep']
    test.loc[mask, i] = 0

df = test.select_dtypes(exclude='object')
nan = df.isna().sum()[df.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with other Cost Nan values:

imputer_median = SimpleImputer(strategy='median')
for i in Expences:
    test[i] = imputer_median.fit_transform(test[[i]])

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Getting a new column of Total Cost:

test['Total_Spend'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']

# %% Let's Check the Relation with Spending and CryoSleep:

rel_cryo = test[test['CryoSleep'].isna()]

'''
We found out that if Total_Spending != 0, that person isn't sleeping.
'''

test['CryoSleep'] = test.apply(lambda x: False if x['Total_Spend'] != 0 else True, axis=1)

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with Age Nan with median values:

test['Age'] = imputer_median.fit_transform(test[['Age']])

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / data.shape[0]) * 100)

# %% Dealing with the Nan values with column of Cabin:

mode = lambda x: x.mode().iloc[0] if not x.mode().empty else None

test.loc[:, 'Cabin_Side'] = test.loc[:, 'Cabin_Side'].fillna(test.groupby('Destination')['Cabin_Side'].transform(mode))
test.loc[:, 'Cabin_Deck'] = test.loc[:, 'Cabin_Deck'].fillna(test.groupby('Destination')['Cabin_Deck'].transform(mode))

test.loc[:, 'Cabin_Side'] = test.loc[:, 'Cabin_Side'].fillna(test.groupby('VIP')['Cabin_Side'].transform(mode))
test.loc[:, 'Cabin_Deck'] = test.loc[:, 'Cabin_Deck'].fillna(test.groupby('VIP')['Cabin_Deck'].transform(mode))

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Dealing with the Nan values of the column of VIP:

mode = lambda x: x.mode().iloc[0] if not x.mode().empty else None

test.loc[:, 'VIP'] = test.loc[:, 'VIP'].fillna(test.groupby('Cabin_Side')['VIP'].transform(mode))

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Dealing with Nan values of Destination:

test.loc[:, 'Destination'] = test.loc[:, 'Destination'].fillna(test.groupby('Cabin_Side')['Destination'].transform(mode))

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Dealing with Nan values:

test.loc[:, 'HomePlanet'] = test.loc[:, 'HomePlanet'].fillna(test.groupby('Cabin_Side')['HomePlanet'].transform(mode))

nan = test.isna().sum()[test.isna().sum() != 0].sort_values(ascending=False)
print((nan / test.shape[0]) * 100)

# %% Getting the Dependent and Independent Variables:

data['CryoSleep'] = data['CryoSleep'].astype('object')
test['CryoSleep'] = test['CryoSleep'].astype('object')

X_train = data.drop('Transported', axis=1)
y_train = data['Transported']
X_test = test

# %% Encoding the categorical data:

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

print(X_train.info())
print(X_test.info())

# %% Scalling the numerical values:

minmax = MinMaxScaler()
cols = X_train.select_dtypes(include='float64').columns
for i in cols:
    X_train.loc[:, i] = minmax.fit_transform(X_train.loc[:, [i]])
    X_test.loc[:, i] = minmax.fit_transform(X_test.loc[:, [i]])

# %% Prediction:

xgb_r = xg.XGBClassifier(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)
xbr1 = xgb_r.score(X_train, y_train)
print(xbr1)

Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
#Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
rf1 = Rf.score(X_train, y_train)
print(rf1)

# %%

Submission = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\sample_submission.csv")
Submission['Transported'] = Y_predict
Submission['Transported']=Submission['Transported']>0.5
Submission.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/sub.csv", index=False)


# %%
ex = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\submission (1).csv")
print(ex.info())

