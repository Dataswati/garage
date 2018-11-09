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
    assert not (input_columns is None), "input_columns is not defined"
    assert not (model is None), "model is not defined"
    x = np.array([float(dict_values[col]) for col in input_columns])
    x = x.reshape(1,-1)
    y_pred = model.predict(x)[0]
    return y_pred
