import pandas as pd
import geopandas
from scipy.spatial import cKDTree
from shapely.geometry import Point
from geopandas import gpd
from geoalchemy2.types import Geometry

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from xgboost import XGBRFClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import getpass


class ModelCompetition:
    def __init__(self, train_set, test_set, tunnel=False):
        self.train_set = train_set
        self.test_set = test_set
        self.response_variable = "is_Coral/Algae"

        self.normalize = True
        self.scores = []
        self.models = {}

        self.X_train = self.train_set.loc[:, self.train_set.columns != self.response_variable]
        self.X_train_norm = (self.X_train-self.X_train.mean())/self.X_train.std()
        self.y_train = self.train_set[self.response_variable]

        self.X_test = self.test_set.loc[:, self.test_set.columns != self.response_variable]
        self.X_test_norm = (self.X_test-self.X_test.mean())/self.X_test.std()
        self.y_test = self.test_set[self.response_variable]

        if tunnel:
            server = SSHTunnelForwarder(
                ('pace-ice.pace.gatech.edu', 22),
                ssh_username=input("Username: "),
                ssh_password=getpass.getpass(prompt='Password: ', stream=None),
                remote_bind_address=('127.0.0.1', 5432)
            )
            server.start()
            local_port = str(server.local_bind_port)
            engine = create_engine('postgresql://{}@{}:{}/{}'.format("jmattingly31", "127.0.0.1",
                                                                     local_port, "coral_data"))
        else:
            engine = create_engine('postgresql://jmattingly31@localhost:5432/coral_data')

    def random_forest(self):
        return 0

    def xgboost_trees(self):
        return 0

    def logistic_regression(self):
        return 0

    def svm(self):
        return 0

    def run_competition(self):
        model_rf = self.random_forest()
        model_xgbt = self.xgboost_trees()
        model_lr = self.logistic_regression()
        model_svm = self.svm()
        for model in [model_rf, model_xgbt, model_lr, model_svm]:
            self.scores.append(accuracy_score(self.y_test, model))


if __name__ == "__main__":
    pass
