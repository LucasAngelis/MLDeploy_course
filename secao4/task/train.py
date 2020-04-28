import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df=data, target=config.TARGET)


# get first letter from cabin variable
X_train[config.IMPUTATION_DICT["cabin_col"]] = pf.extract_cabin_letter(df=X_train, 
	var=config.IMPUTATION_DICT["cabin_col"])

# impute categorical variables
for cat_var in config.CATEGORICAL_VARS:
	X_train[cat_var] = pf.impute_na(df=X_train, var=cat_var, value_for_replacing=None)

# impute numerical variable
for num_var in config.NUMERICAL_TO_IMPUTE:
	X_train[num_var+'_NA'] = pf.add_missing_indicator(df=X_train, var=num_var)
	X_train[num_var] = pf.impute_na(df=X_train, var=num_var, 
		value_for_replacing=config.IMPUTATION_DICT['median'][num_var])

# Group rare labels
for var in config.FREQUENT_LABELS.keys():
    X_train[var] = pf.remove_rare_labels(df=X_train, var=var,
        frequent_labels=config.FREQUENT_LABELS[var])


# encode categorical variables
X_train_ohe = pf.encode_categorical(df=X_train, var=config.CATEGORICAL_VARS)



# check all dummies were added
X_train_ohe_check = pf.check_dummy_variables(df=X_train_ohe, dummy_list=config.DUMMY_VARIABLES)


# train scaler and save
scaler = pf.train_scaler(df=X_train_ohe_check, output_path=config.OUTPUT_SCALER_PATH)


# scale train set
X_train_ohe_check_scale = pf.scale_features(df=X_train_ohe_check,
	output_path=config.OUTPUT_SCALER_PATH)


# train model and save
model = pf.train_model(df=X_train_ohe_check_scale, target=y_train,
	output_path=config.OUTPUT_MODEL_PATH)


print('Finished training')