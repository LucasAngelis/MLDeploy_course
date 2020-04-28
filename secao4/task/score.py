import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # extract first letter from cabin
    data[config.IMPUTATION_DICT["cabin_col"]] = pf.extract_cabin_letter(df=data, 
        var=config.IMPUTATION_DICT["cabin_col"])

    # impute NA categorical
    for cat_val in config.CATEGORICAL_VARS:
        data[cat_val] = pf.impute_na(df=data, var=cat_val, value_for_replacing=None)
    
    
    # impute NA numerical
    for num_val in config.NUMERICAL_TO_IMPUTE:
        data[num_val+'_NA'] = pf.add_missing_indicator(df=data, var=num_val)
        data[num_val] = pf.impute_na(df=data, var=num_val,
            value_for_replacing=config.IMPUTATION_DICT['median'][num_val])
    
    
    # Group rare labels
    for var in config.FREQUENT_LABELS.keys():
        data[var] = pf.remove_rare_labels(df=data, var=var,
            frequent_labels=config.FREQUENT_LABELS[var])
    
    # encode variables
    data_ohe = pf.encode_categorical(df=data, var=config.CATEGORICAL_VARS)
        
        
    # check all dummies were added
    data_ohe_check = pf.check_dummy_variables(df=data_ohe, dummy_list=config.DUMMY_VARIABLES)
    
    # scale variables
    data_ohe_check_scale = pf.scale_features(df=data_ohe_check, output_path=config.OUTPUT_SCALER_PATH)
    
    # make predictions
    predictions = pf.predict(df=data_ohe_check_scale, model_path=config.OUTPUT_MODEL_PATH)

    
    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        