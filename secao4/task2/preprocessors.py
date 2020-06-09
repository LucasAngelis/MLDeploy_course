import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        # store numerical data columns names
        self.num_cols = variables


    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self


    def transform(self, X):
        # add indicator
        X = X.copy()
        # iterate through columns
        for col_name in self.num_cols:
            # Add new col with '1' indicating null value
            X[col_name + '_NA'] = np.where(X[col_name].isnull(), 1, 0)

        return X
        


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        # store categorical data columns names
        self.cat_cols = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        # iterate through columns
        for col_name in self.cat_cols:
            # Add new col with 'Missing' on missing values
            X[col_name] = X[col_name].fillna('Missing')

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        # store numerical data columns names
        self.num_cols = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        # iterate through columns
        for col_name in self.num_cols:
            # Save median value into imputer_dict
            self.imputer_dict_[col_name] = X[col_name].median()

        return self

    def transform(self, X):
        X = X.copy()
        # iterate through columns
        for col_name in self.num_cols:
            # input median value into null ones
            X[col_name].fillna(self.imputer_dict_[col_name], inplace=True)

        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.var = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        # iterate through columns
        for col_name in self.var:
            # get first letter from string
            X[col_name] = X[col_name].str[0]

        return X

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        self.var = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        # iterate through columns
        for col_name in self.var:
            # Group by category and calculate frequency
            tmp = X.groupby(col_name)[col_name].count() / len(X)
            # store rare labels names into encoder dict
            self.encoder_dict_[col_name] = tmp[tmp > self.tol].index

        return self

    def transform(self, X):
        X = X.copy()
        # iterate through columns
        for col_name in self.var:
            # Replace rare label for the string 'Rare'
            X[col_name] = np.where(X[col_name].isin(self.encoder_dict_[col_name]),
             X[col_name], 'Rare')

        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.var = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.var], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        for col_name in self.var:
            X = pd.concat([X,
                pd.get_dummies(X[col_name], prefix=col_name, drop_first=True)], 
                axis=1)
        # drop original variables
        X.drop(labels=self.var, axis=1, inplace=True)
        # add missing dummies if any
        missing_cols = list(set(self.dummies) - set(X.columns))
        if missing_cols:
            for col_name in missing_cols:
                X[col_name] = 0

        return X
