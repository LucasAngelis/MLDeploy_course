import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import titanic_pipe
import config



def run_training():
    """Train the model."""

    # read training data
    df = pd.read_csv(config.TRAINING_DATA_FILE)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
	    df.drop(config.TARGET, axis=1),  # predictors
	    df[config.TARGET],  # target
	    test_size=0.2,  # percentage of obs in test set
	    random_state=0)  # seed to ensure reproducibility
    # fit pipeline
    titanic_pipe.fit(X_train, y_train)
    # save pipeline
    joblib.dump(titanic_pipe, config.PIPELINE_NAME)

if __name__ == '__main__':
    run_training()
