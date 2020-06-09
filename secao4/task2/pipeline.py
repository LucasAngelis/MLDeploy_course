from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
   	[
	    ('get_first_letter', 
	    	pp.ExtractFirstLetter(variables=config.CABIN)),
	    ('numerical_missing_indicator', 
	    	pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
	    ('numerical_missing_inputer',
	    	pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
	    ('categorical_missing_inputer',
	    	pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
	    ('rare_label_encoder',
	    	pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
	    ('one_hot_encoder',
	    	pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
	    ('numerical_scaler', StandardScaler()),
	    ('logistic_regressor', LogisticRegression(C=0.0005, random_state=0))
    ]
)