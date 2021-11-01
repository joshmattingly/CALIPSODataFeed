import pandas as pd
import geopandas

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from xgboost import XGBRFClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


class ModelCompetition:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.response_variable = "is_Coral/Algae"

        self.normalize = True
        self.test_size = 0.3
        self.scores = []
        self.models = {}

    def accuracy_scores(self):
        pass

    def random_forest(self):
        pass

    def xgboost_trees(self):
        pass

    def logistic_regression(self):
        pass

    def svm(self):
        pass

    def run_competition(self):
        pass