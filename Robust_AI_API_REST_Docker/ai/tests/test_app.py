from flask import url_for

def test_my_json_response(client):
    res = client.get(url_for('API./_train'))
    assert res.status_code == 200,res.status_code
