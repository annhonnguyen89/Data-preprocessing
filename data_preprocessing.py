# -*- coding: utf-8 -*-

from sklearn import ensemble
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import pandas as pd
from scipy import stats
import numpy as np

datafold = ""
train_file = ""
test_file = ""


def remove_constant_variables(df_train, df_test):
    '''
    creat DataFrame from file
    remove columns with constant value
    '''
    
    print("original size")
    print(df_train.shape)
    
    missing_df = df_train.isnull().sum(axis = 0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by = 'missing_count')
    
    df_unique = df_train.nunique().reset_index()
    df_unique.columns = ['column name', 'unique values']
    
    # remove columns with only 1 value
    df_train1 = df_train[df_unique.loc[df_unique['unique values'] > 1, "column name"]]
    tmp = df_unique.loc[df_unique['unique values'] > 1, "column name"].values.tolist()
    X_col_names = [x for x in tmp if x not in ["target"]]
    df_test1 = df_test[X_col_names]
    return(df_train1, df_test1)

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    '''
    predict by Light GBM
    '''
    params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 30,
            "learning_rate": 0.01,
            "bagging_fraction": 0.7,
            #"feature_fraction": 5, 
            "bagging_frequency": 5,
            "bagging_seed": 2018,
            "verbosity": -1
            }

    lgtrain = lgb.Dataset(train_X, label = train_y)
    lgval = lgb.Dataset(val_X, label = val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets = [lgval], early_stopping_rounds = 100, 
                                                           verbose_eval= 200, evals_result = evals_result)
    pred_test_y = model.predict(test_X, num_iteration = model.best_iteration)
    return pred_test_y, model, evals_result  
  
def modeling(df_train, df_test):
    '''
    modelling using KFold for validation, Light GBM
    '''
    df_test = df_test[0:10]
    train_X = df_train.drop(["ID", "target"], axis = 1)
    train_Y = np.log1p(df_train['target'].values)
    
    test_X = df_test.drop(["ID"], axis = 1)
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state = 231)
    pred_test_full = 0
    
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index, :]
        dev_y, val_y = train_Y[dev_index], train_Y[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
        pred_test_full += pred_test
    
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)

    sub_df = pd.DataFrame({"ID":df_test["ID"].values})
    sub_df["target"] = pred_test_full
    sub_df.to_csv(datafold + "\\result.csv", index=False)

def check_balance(df_train, df_test):
    '''
    remove variables with only 2 (or 3 values) and the frequency of 1 value < 0.001 (or 2 values less than 5)
    '''
    df_unique = df_train.nunique().reset_index()
    df_unique.columns = ["column name", "unique values"]
    cat_cols = df_unique.loc[df_unique["unique values"] < 4, "column name"].values.tolist()
        
    for col in cat_cols:
        vals = df_train[col]
        freq = vals.value_counts()
        # remove binary variables with unbalanced values
        if(len(freq) == 2 and (freq[freq.index[0]]/freq[freq.index[1]] < 0.001 or freq[freq.index[1]]/freq[freq.index[0]] < 0.001)):
            df_train = df_train.drop(col, axis  = 1)
            df_test = df_test.drop(col, axis = 1)
            
        elif(len(freq) == 3 and sum([(freq[freq.index[0]] < 5), (freq[freq.index[1]] < 5), (freq[freq.index[2]] < 5)]) == 2):
            df_train = df_train.drop(col, axis  = 1)
            df_test = df_test.drop(col, axis = 1)
            
    return([df_train, df_test])
          
def encode_category(df_train, df_test):
    '''
    columns with less than 4 value --> encode to category variables
    '''
    df_unique = df_train.nunique().reset_index()
    df_unique.columns = ["column name", "unique values"]
    cat_cols = df_unique.loc[df_unique["unique values"] < 4,"column name"].values.tolist()
    
    df_train = pd.get_dummies(df_train, columns = cat_cols, prefix = cat_cols)
    df_test = pd.get_dummies(df_test, columns = cat_cols, prefix = cat_cols)
    return([df_train, df_test])

def transformation(df_train, df_test):
    # transform variable if distrbution is skewed
    '''
    if variable is negative skewed (<-0.5) --> square
    if variable is positive skewed (>0.5) --> log it
    '''
    cols = [x for x in df_train.columns.tolist() if x not in ["ID", "target"]]
    for col in cols:
        if(len(df_train[col].value_counts()) > 3 ):
            sk = stats.skew(df_train[col])
            if (sk < -0.5):
                df_train[col] = np.power(df_train[col], 2)
                if(col in df_test.columns.tolist()):
                    df_test[col] = np.power(df_test[col], 2)
            elif (sk > 0.5):
                df_train[col] = np.array([np.log1p(x + 1) for x in df_train[col]])
                if(col in df_test.columns.tolist()):
                    df_test[col] = np.array([np.log1p(x + 1) for x in df_test[col]])
            
    return([df_train, df_test])


def feature_importance(df_train, df_test):
    '''
    compute variable importance by ExtraTreeRegressor
    '''
    train_X = df_train.drop(["ID", "target"], axis = 1)
    train_Y = np.log1p(df_train['target'].values)
    
    test_X = df_test.drop(["ID", "target"], axis = 1)

    model = ensemble.ExtraTreesRegressor(n_estimators = 200, max_depth = 20, 
                                      max_features = 0.5, n_jobs = -1, random_state = 0)
    model.fit(train_X, train_Y)
    feat_names = train_X.columns.values
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1][:20]
    print(importances.shape)
    
    plt.figure(figsize = (20,20))
    plt.title("feature importances")
    plt.bar(range(len(indices)), importances[indices], color ="r", yerr = std[indices], align = "center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation = "vertical")
    
    plt.xlim([-1, len(indices)])
    plt.show()
    
    return()

if __name__ == "__main__":
    df_train = pd.read_csv(datafold + "\\" + train_file)
    df_test = pd.read_csv(datafold + "\\" + test_file)        
    [df_train, df_test] = remove_constant_variables(df_train, df_test)
    [df_train, df_test] = check_balance(df_train, df_test)
    [df_train, df_test] = check_outliers(df_train, df_test)
    modeling(df_train, df_test)
