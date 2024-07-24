"""Gets data from web using webscrapper."""

# %% IMPORTS

from abc import ABC, abstractmethod
import typing as T
import pydantic as pdt
import pandas as pd
from abc import ABC, abstractmethod
from loguru import logger


from sklearn import ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

# %% READERS

class Training(ABC, pdt.BaseModel, strict=True, frozen=True, extra='forbid'):
    """
    Base class to read data from dataset.

    Used to get data from the dataset and load as dataframe into memory.

    Parameters:
        limit (int, optional): maximum number of rows to read. Defautls to none.
    """
    
    KIND: str

    @abstractmethod
    def run(self, data: pd.DataFrame):
        """"
        Reads a dataframe from a dataset.

        Returns:
            pd.Dataframe: dataframe representation.
        """


class TrainingData(Training, strict=True, frozen=True, extra="forbid"):
    """
    Reads a dataframe from the NFL provided website.
    """

    KIND: T.Literal["TrainingData"] = "TrainingData"

    @T.override
    def run(self, data: pd.DataFrame):
        data['team_a_win'] = [True if x == 'team_a' else False for x in data['outcome']]

        features = [
            'team_a_yds',
            'team_a_turnovers',
            'team_b_yds',
            'team_b_turnovers',
            'pts_per_year_team_a',
            'pts_per_year_team_b'
        ]

        X = data[features]
        y = data['team_a_win']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        arvore = ensemble.RandomForestClassifier(random_state=841)
        param_grid = {
            'n_estimators': [100, 200,300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        grid_search = GridSearchCV(estimator=arvore, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train,y_train)
        logger.info("Best parameters found: {}", grid_search.best_params_)
        best_arvore = grid_search.best_estimator_

        y_pred = best_arvore.predict_proba(X_test)[:, 1]

        valid_score = roc_auc_score(y_test, y_pred)
        logger.info("Validation ROC-AUC score: {}", valid_score)

        valid_accuracy = accuracy_score(y_test, y_pred > 0.5)
        logger.info("Accuracy of P>0.5 classifier: {}", valid_accuracy)
        
        return best_arvore

TrainingKind = TrainingData