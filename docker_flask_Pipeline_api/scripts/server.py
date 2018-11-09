import os
from flask import Flask, jsonify, request, session

import json
from prediction import predict, train
import joblib


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

dumped_pipeline = "/srv/dump_pipeline.dump"
folder_csv_path = "/srv/csv/"

class DataStore():
    is_train = None
    input_columns = None

data = DataStore()


def flask_app():
    app = Flask(__name__)
    app.secret_key = 'there is not secret ai'

    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up'

    @app.route('/available_input', methods=['GET'])
    def get_input():
        # is_train = session.get('is_train',False)=="True"
        is_train = data.is_train
        if is_train :
            # return jsonify({'status':'OK','input_names':session.get('input_columns',None)})
            return jsonify({'status':'OK','input_names':data.input_columns})
        else :
            return jsonify({'status':'KO', 'comment':'not trained yet'}), 401


    @app.route('/train', methods=['POST'])
    def train_route():
        input_args = request.json
        if not ("target" in input_args.keys()):
            return jsonify({'status':'KO', 'comment':'need a target to predict'}), 401
        target = input_args["target"]
        if not ("csv_file_name" in input_args.keys()):
            return jsonify({'status':'KO', 'comment':'need a csv file_name'}), 401
        csv_file_name = input_args["csv_file_name"]
        csv_path = os.path.join(folder_csv_path, csv_file_name)

        if not ("input_columns" in input_args.keys()):
            input_columns = None
        else :
            input_columns = input_args["input_columns"]
        best_test_score, best_pipeline, input_columns = train(csv_path,target=target,input_columns=input_columns)
        # session["input_columns"] = input_columns
        data.input_columns = input_columns
        joblib.dump(best_pipeline, dumped_pipeline)
        # session["is_train"]="True"
        data.is_train = True
        return jsonify({"status":'OK', 'test_score':best_test_score})

    @app.route('/predict', methods=['POST'])
    def pred_route():
        to_predict = request.json

        print(to_predict)
        # is_train = session.get('is_train',False)=="True"
        is_train = data.is_train

        if not is_train :
            return jsonify({'status':'KO', 'comment':'not trained yet'}), 401
       	 
        # input_columns = session.get('input_columns',None)
        input_columns = data.input_columns

        pipeline = joblib.load(dumped_pipeline)

        pred = predict(to_predict, input_columns=input_columns, model=pipeline)

        return jsonify({"status":"OK", "predict cost":pred})
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')


