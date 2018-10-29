import os
from flask import Flask, jsonify, request
# from PowerOP.interfaces.Diag import InterfaceDiag

import json
from prediction import prediction


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def flask_app(url):
    app = Flask(__name__)


    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up'

    @app.route('/predict_price', methods=['POST'])
    def start():
        to_predict = request.json

        print(to_predict)
	pred = prediction(to_predict)
        return "predict cost : %s"%(pred,)
    return app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


