from flask import url_for
import time

def test_train_predict_route(client):
    if False :
        res = client.delete(url_for('API./_train'))
        assert res.status_code == 200,res.status_code

        res=client.post(url_for('API./_train'),
                        json={"target":"price",
                              "csv_file_name":"kc_house_data.csv"})

        while(True):
            res = client.get(url_for('API./_train'))
            # print(res.json)
            if not res.json['training']:
                break
            time.sleep(5)


        
    res = client.get(url_for('API./_train'))
    assert res.status_code == 200,res.status_code
    assert res.json['training']==False,res.json['training']
    assert res.json['model_available']==True,\
            res.json['model_available']

    res=client.post(url_for('API./_train'),
                    json={"target":"price",
                          "csv_file_name":"kc_house_data.csv"})
    assert res.status_code == 200,res.status_code
    print(res.json)

    res = client.get(url_for('API./_train'))
    assert res.status_code == 200, res.status_code
    assert res.json['training']==True, res.json['training']
    assert res.json['model_available']==True,\
            res.json['model_available']

    res = client.get(url_for('API./_predict'))
    print(res.json)
    json_predict_string = open('/data/to_predict_json.json').read()
    json_predict = json.loads(json_predict_string)
    res = client.post(url_for('API./_predict'),json=json_predict)
    assert res.status_code == 200,res.status_code
    assert prediction in res.json.keys(), res.json


