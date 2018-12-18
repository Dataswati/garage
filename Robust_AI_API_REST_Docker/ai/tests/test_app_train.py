from flask import url_for
import time

def test_train_route(client):
    res = client.delete(url_for('API./_train'))
    res = client.get(url_for('API./_train'))
    assert res.status_code == 200,res.status_code
    assert res.json['training']==False,res.json['training']
    assert res.json['model_available']==False,\
            res.json['model_available']

    res=client.post(url_for('API./_train'),
                    json={"target":"price",
                          "csv_file_name":"kc_house_data.csv"})#,
                    # headers={"content-type" : "application/json"})

    assert res.status_code == 200, res.status_code
    time.sleep(2)
    res = client.get(url_for('API./_train'))
    assert res.status_code == 200, res.status_code
    assert res.json['training']==True, res.json['training']
    assert res.json['model_available']==False,\
            res.json['model_available']

    res = client.delete(url_for('API./_train'))
    res = client.get(url_for('API./_train'))
    assert res.status_code == 200,res.status_code
    assert res.json['training']==False,res.json['training']
    assert res.json['model_available']==False,\
            res.json['model_available']

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
    print(res.json)
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
