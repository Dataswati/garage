import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, QuantileTransformer

Pipelines = [
    ("gbr", GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')),
    ("st_lasso", Pipeline([('st', StandardScaler()), ('Lasso', Lasso())])),
    ("st_sfm_svr", Pipeline([('st', QuantileTransformer()), ('SFM', SelectFromModel(Lasso())),
                             ('SVR', SVR(C =100, epsilon=10000))])),
    ("sfm_gbr", Pipeline([('SFM', SelectFromModel(RandomForestRegressor(n_estimators=400, max_depth=5,
                                                                        min_samples_split=2),threshold=0,max_features=7)), 
              ('GBR', GradientBoostingRegressor(n_estimators=400, max_depth=5,
                                                min_samples_split=2, learning_rate=0.1, loss='ls'))]))
]

def train(path_csv_train, target=None, input_columns=None, Pipelines=Pipelines):
    """
    train on the specified csv file with the specified target

    Parameters 
    ---------
    csv_file_name : str
        the path of the csv file use for the training
    target : str
        the target to predict present as a column in the csv       
    input_columns : str
        the list of the columns that should be used as the input, if not specified
        all the columns execpt the target will be used

    Returned  
    ------------------
    best_test_score: float
         R2 score of the best model on a test set

    best_pipeline: Pipeline or model sklearn object
         the pipeline that gived the best test score
    input_columns:
         the allowed input columns (only numeric columns are used)
    """
    assert isinstance(target,str), "target is not defined"
    data = pd.read_csv(path_csv_train)
    data = data.select_dtypes(include=[np.number])
    assert target in data.columns, "target should be in the columns list ; %s"%(data.columns,)
    if input_columns is None :
        input_columns = list(data.columns)
        input_columns.remove(target)
    
    X = data.drop(target,axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2)
    list_test_score = []
    for pipeline_name, pipeline in Pipelines :
        pipeline.fit(x_train, y_train)
        test_score = pipeline.score(x_test, y_test)
        print("%s test score : %s"%(pipeline_name, test_score))
        list_test_score.append(test_score)
    best_test_score = np.max(list_test_score)
    best_pipeline = Pipelines[np.argmax(list_test_score)][1]
    return best_test_score, best_pipeline, input_columns

def predict(dict_values, input_columns=None, model=None):
    """
    predict the value from a dict

    Parameters 
    ----------
    dict_values: dict
        A dict that contain columns name as key and input as value
    input_columns: list
        A exaustive list of allowed values that should be present in
        the dict keys

    Return 
    ------
    y_pred : float
        the prediction 
    """
    assert not (input_columns is None), "input_columns is not defined"
    assert not (model is None), "model is not defined"
    x = np.array([float(dict_values[col]) for col in input_columns])
    x = x.reshape(1,-1)
    y_pred = model.predict(x)[0]
    return y_pred
