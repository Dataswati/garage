from model import predict, train
import os
import pytest
import tempfile
import pandas as pd

df1 = pd.read_csv("/data/kc_house_data.csv")

df2 = df1.copy()
df2 = df1.iloc[:1000]

@pytest.mark.parametrize("target",["price", "sqft_living"])
@pytest.mark.parametrize("df",[df1, df2])
@pytest.mark.parametrize("input_columns",[None,["sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated"]])



def test_train_pred(target, df,input_columns):
    with tempfile.TemporaryDirectory() as temp_dir:
       path_csv = os.path.join(temp_dir, "test.csv") 
       df.to_csv(path_csv,index=False)
       best_test_score, best_pipeline, input_columns = \
           train(path_csv, target=target,input_columns=input_columns)
       assert isinstance(input_columns, list)
       dict_input = df[input_columns].iloc[42].T.to_dict()
       pred = predict(dict_input, model=best_pipeline, input_columns=input_columns)
       assert isinstance(pred,float)
