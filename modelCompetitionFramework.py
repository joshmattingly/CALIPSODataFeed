from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from xgboost import XGBRFClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sqlalchemy import create_engine
# from sshtunnel import SSHTunnelForwarder
# import getpass


class ModelCompetition:
    def __init__(self, data, tunnel=False, manta=False, manta_dist=0.5, manta_coral_thresh=10, sample_size=6, seed=42):
        self.response_variable = "is_Coral/Algae"
        self.seed = seed
        self.normalize = True
        self.scores = []
        self.models = {}
        self.importances = None
        self.std = []
        self.clf = None
        # TODO: Convert train/test initializers to postgis tables
        '''
        username = input("Username: ")
        password = getpass.getpass(prompt='Password: ', stream=None)
        if tunnel:
            server = SSHTunnelForwarder(
                ('pace-ice.pace.gatech.edu', 22),
                ssh_username=username,
                ssh_password=password,
                remote_bind_address=('127.0.0.1', 5432)
            )
            server.start()
            local_port = str(server.local_bind_port)
            engine = create_engine('postgresql://{}@{}:{}/{}'.format(username, "127.0.0.1",
                                                                     local_port, "coral_data"))
        else:
            engine = create_engine('postgresql://{}@localhost:5432/coral_data'.format(username))
        '''
        # TODO: Manta tow coral regression algorithms
        # TODO: Add PCA transformation

        if manta:
            self.manta_dist = manta_dist
            self.manta_coral_tresh = manta_coral_thresh
            self.sample_size = sample_size
            # create boolean flag based on distance
            # create boolean flag based on average number of coral
            data = data.assign(y=np.where((data['dist'] <= self.manta_dist)
                                                 & (data['mean_live_coral'] >= self.manta_coral_tresh), 1, 0))

            data.drop(['Date',
                       'Long',
                       'Lat',
                       'geom',
                       'sample_date',
                       'sector',
                       'shelf',
                       'reef_name',
                       'reef_id',
                       'p_code',
                       'visit_no',
                       'median_live_coral',
                       'median_soft_coral',
                       'median_dead_coral',
                       'mean_live_coral',
                       'mean_dead_coral',
                       'total_cots',
                       'mean_cots_per_tow',
                       'tows',
                       'dist'], axis=1, inplace=True)
            nrows = len(data)
            data = data.groupby('y').apply(lambda x: x.sample(n=int(nrows/self.sample_size))).reset_index(drop=True)

            self.X = data.loc[:, data.columns != 'y']
            self.X_norm = (self.X-self.X.mean())/self.X.std()
            self.y = data['y']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y,
                                                                                    test_size=0.3,
                                                                                    random_state=self.seed)
        else:
            data.columns = ['y' if x=='is_Coral/Algae' else x for x in data.columns]
            self.X = data.loc[:, data.columns != 'y']
            self.X_norm = (self.X-self.X.mean())/self.X.std()
            self.y = data['y']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_norm, self.y,
                                                                                    test_size=0.3, random_state=self.seed)

    def random_forest(self):
        self.clf = RandomForestClassifier(random_state=self.seed)
        self.clf.fit(self.X_train, self.y_train)
        y_hat = self.clf.predict(self.X_test)
        self.importances = self.clf.feature_importances_
        self.std = np.std([tree.feature_importances_ for tree in self.clf.estimators_], axis=0)
        return y_hat

    def xgboost_forest(self):
        model = XGBRFClassifier(n_estimators=100)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        return y_hat

    def run_competition(self):
        model_rf = self.random_forest()
        model_xgrf = self.xgboost_forest()

        for model in [model_rf, model_xgrf]:
            self.scores.append(accuracy_score(self.y_test, model))


if __name__ == "__main__":
    print("Importing Florida Data")
    gdf_class_test = pd.read_csv('florida_200_classified.csv')
    gdf_class_test = gdf_class_test[gdf_class_test['class'] != 'Seagrass']
    print("Creating dummy variables")
    dummies = gdf_class_test['class'].str.get_dummies()
    dummies.columns = ['is_' + col for col in dummies.columns]
    print("Creating test frame")
    gdf_class_test = pd.concat([gdf_class_test, dummies], axis=1)
    print("Creating baseline frame")
    gdf_chlor = gdf_class_test[['chlorophyll', 'is_Coral/Algae']]

    gdf_class_test.drop(['index', 'Unnamed: 0', 'calipso_date', 'Date', 'Date.1', 'Long', 'Lat', 'geom', 'geometry',
                         'dist', 'neo_file_date', 'Latitude', 'Longitude', 'is_Rock', 'is_Rubble', 'is_Sand', 'class',
                         'chlorophyll'],
                        axis=1, inplace=True)

    # Baseline Test (predictions based only on chlorophyll concentrations
    print("Running baseline competition")
    modelChlor = ModelCompetition(gdf_chlor)
    modelChlor.run_competition()
    print(modelChlor.scores)
    # [RF: 0.9066769367981514, XGBF: 0.7470709855272226]
    print("Running actual competition")

    bin_starts = list(reversed(range(0, 200, 25)))
    acc_collector = []
    for n in bin_starts:
        if n == 0:
            n = 1
        print("Running competitions for n = {}".format(n))
        gdf_subset = pd.concat([gdf_class_test[['Land_Water_Mask']], gdf_class_test.iloc[:, n:]], axis=1)
        modelComp = ModelCompetition(gdf_subset)
        modelComp.run_competition()
        print(modelComp.scores)
        acc_collector.append([n, modelComp.scores])
        print("Calculating Feature Importance")
        result = permutation_importance(
            modelComp.clf, modelComp.X_test, modelComp.y_test, n_repeats=10, random_state=42, n_jobs=-1
        )

        forest_importances = pd.Series(result.importances_mean, index=modelComp.clf.feature_names_in_)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
        fig.set_size_inches(8, 6)
        fig.savefig("all_features - {}.png".format(n), dpi=100)

        top_features = forest_importances.nlargest(5)
        fig, ax = plt.subplots()
        top_features.plot.bar(ax=ax)
        ax.set_title("Top CALIPSO Features")
        ax.set_ylabel("Mean decrease in impurity")
        fig.set_size_inches(8, 6)
        fig.savefig("top_features - {}.png".format(n), dpi=100)
    print(acc_collector)

    print("Running GBR Test")
    gdf_manta_test = pd.read_csv('coral_data_public_gbr_200.csv')
    modelGBR = ModelCompetition(gdf_manta_test, manta=True)
    modelGBR.run_competition()
