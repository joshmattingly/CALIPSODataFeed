import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from xgboost import XGBRFClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


data = pd.read_csv('florida_full.csv')
data = data[data.benth_dist <= 0.5]
data = data[data.benth_class != 'Seagrass']

dummies = data.benth_class.str.get_dummies()
dummies.columns = ['is_' + col for col in dummies.columns]
data = pd.concat([data, dummies], axis=1)

geo_dummies = data.geo_class.str.get_dummies()
geo_dummies.columns = ['is_' + col for col in geo_dummies.columns]
data = pd.concat([data, geo_dummies], axis=1)

data.drop(['Unnamed: 0', 'calipso_date', 'neo_file_date', 'geometry', 'neo_date', 'neo_dist', 'benth_dist',
           'geo_class', 'geo_dist', 'benth_class', 'is_Rock', 'is_Rubble', 'is_Sand', 'is_Back Reef Slope',
           'is_Inner Reef Flat', 'is_Outer Reef Flat', 'is_Plateau', 'is_Reef Crest', 'is_Reef Slope',
           'is_Shallow Lagoon', 'is_Sheltered Reef Slope', 'is_Terrestrial Reef Flat', 'is_Deep Lagoon'],
          axis=1, inplace=True)


# Use this to split entire dataset
X = data.loc[:, data.columns != 'is_Coral/Algae']
X_norm = (X-X.mean())/X.std()
y = data['is_Coral/Algae']

# Another option is to use this to sample a close to 50% split between coral/algae True and coral/algae False
# sample_space = data.groupby('is_Coral/Algae').apply(lambda x: x.sample(33000))
# X = sample_space.loc[:, sample_space.columns != 'is_Coral/Algae']
# y = sample_space['is_Coral/Algae']
train, test = train_test_split(data, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

forest_scores = cross_val_score(clf, X_norm, y, scoring='accuracy', cv=cv, n_jobs=-1)

# XGBoost
model = XGBRFClassifier(n_estimators=100)
xg_scores = cross_val_score(model, X_norm, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Traditional Random Forest Mean Accuracy: %.3f (%.3f)' % (mean(forest_scores), std(forest_scores)))
print('XGBoostRandomForest Mean Accuracy: %.3f (%.3f)' % (mean(xg_scores), std(xg_scores)))

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
accuracy_score(y_test, y_hat)
results = pd.DataFrame(list(zip(y_test.to_list(), y_hat)), columns=['y_test', 'y_hat'])
results.to_csv('random_forest_test_2.csv')

X_full = data.loc[:, data.columns != 'is_Coral/Algae']
X_full_norm = (X_full-X_full.mean())/X_full.std()
y_full = data['is_Coral/Algae']
y_hat_full = clf.predict(X_full_norm)
accuracy_score(y_full, y_hat_full)

sns.set_theme(style="darkgrid")
ax = sns.countplot(x="is_Coral/Algae", data=data)