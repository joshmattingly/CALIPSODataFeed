import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from xgboost import XGBRFClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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

data.drop(['Unnamed: 0', 'calipso_date', 'neo_file_date', 'geometry', 'neo_date', 'neo_dist', 'benth_dist',
           'geo_class', 'geo_dist', 'benth_class', 'is_Rock', 'is_Rubble', 'is_Sand'],
          axis=1, inplace=True)


sample_space = data.groupby('is_Coral/Algae').apply(lambda x: x.sample(33000))

X = sample_space.loc[:, sample_space.columns != 'is_Coral/Algae']
X_norm = (X-X.mean())/X.std()
y = sample_space['is_Coral/Algae']
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
forest_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# XGBoost
model = XGBRFClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
xg_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Traditional Random Forest Mean Accuracy: %.3f (%.3f)' % (mean(forest_scores), std(forest_scores)))
print('XGBoostRandomForest Mean Accuracy: %.3f (%.3f)' % (mean(xg_scores), std(xg_scores)))

accuracy_score(y_test, y_hat)
results = pd.DataFrame(list(zip(y_test.to_list(), y_hat)), columns=['y_test', 'y_hat'])
results.to_csv('random_forest_test.csv')