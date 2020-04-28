import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    data = pd.read_csv(df_path)

    return data



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
        df[target],
        test_size=0.2,
        random_state=0)

    return X_train, X_test, y_train, y_test
    


def extract_cabin_letter(df, var):
    # captures the first letter
    new_col = df[var].str[0]

    return new_col



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    new_col = np.where(df[var].isnull(), 1, 0)

    return new_col


    
def impute_na(df, var, value_for_replacing=None):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    if value_for_replacing: # Median value to replace NA
        new_col = df[var].fillna(value_for_replacing)
    else: # Using 'Missing' for categorical
        new_col = df[var].fillna('Missing')

    return new_col




def remove_rare_labels(df, var, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    new_col = np.where(df[var].isin(frequent_labels), df[var], 'Rare')

    return new_col



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    
    df = df.copy()
    # Concat One Hot Encoded variables
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], 
        axis=1)
    # Drop original categorical variable
    df.drop(labels=var, axis=1, inplace=True)

    return df



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    df = df.copy()
    for variable in dummy_list:
        if variable not in df:
            df[variable] = 0

    return df

    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)

    return scaler
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    df_scale = scaler.transform(df)

    return df_scale



def train_model(df, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    model.fit(df, target)
    joblib.dump(model, output_path)

    return model



def predict(df, model_path):
    # load model and get predictions
    model = joblib.load(model_path)
    prediction = model.predict(df)

    return prediction
