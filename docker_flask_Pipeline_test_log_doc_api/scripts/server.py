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
        """a simple route to check if the server is up
        """
        # print("success")
        return 'server is up'

    @app.route('/available_input', methods=['GET'])
    def get_input():
        """
        Return the list of input name. This list should be as key of 
        the json send to the /predict route
        
        Returned json keys
        ------------------
        status : str
             status of the request : OK/KO
        comment : str
             comment that can explain status:KO
        input_names : list
             list of the input for the prediction
        """
        # is_train = session.get('is_train',False)=="True"
        is_train = data.is_train
        if is_train :
            app.logger.info("input names : %s"%(data.input_columns,))
            # return jsonify({'status':'OK','input_names':session.get('input_columns',None)})
            return jsonify({'status':'OK','input_names':data.input_columns})
        else :
            app.logger.bug("/available_input called but not trained yet")
            return jsonify({'status':'KO', 'comment':'not trained yet'}), 401


    @app.route('/train', methods=['POST'])
    def train_route():
        """
        train on the specified csv file with the specified target

        Input json keys 
        ---------
        target : str
            the target to predict present as a column in the csv       
        csv_file_name : str
            the name of the csv file present in de csv folder use for the training
        input_columns : str
            the list of the columns that should be used as the input, if not specified
            all the columns execpt the target will be used

        Returned json keys 
        ------------------
        status : str
             status of the request : OK/KO
        comment : str
             comment that can explain status:KO
        test_score: float
             R2 score of the model on a test set
        """
        input_args = request.json
        if not ("target" in input_args.keys()):
            app.logger.bug("/train need a target to predict")
            return jsonify({'status':'KO', 'comment':'need a target to predict'}), 401
        target = input_args["target"]
        if not ("csv_file_name" in input_args.keys()):
            app.logger.bug("/train need a csv file_name")
            return jsonify({'status':'KO', 'comment':'need a csv file_name'}), 401
        csv_file_name = input_args["csv_file_name"]
        csv_path = os.path.join(folder_csv_path, csv_file_name)

        if not ("input_columns" in input_args.keys()):
            input_columns = None
        else :
            input_columns = input_args["input_columns"]
        app.logger.info("/train start the train")
        best_test_score, best_pipeline, input_columns = train(csv_path,target=target,input_columns=input_columns)
        app.logger.info("/train train finish")
        # session["input_columns"] = input_columns
        data.input_columns = input_columns
        joblib.dump(best_pipeline, dumped_pipeline)
        app.logger.info("/train model dumped")
    
        # session["is_train"]="True"
        data.is_train = True
        return jsonify({"status":'OK', 'test_score':best_test_score})

    @app.route('/predict', methods=['POST'])
    def pred_route():
        """
        predict the value from a json

        Input json keys 
        ---------------
        key:value were key are the input_columns keys that can be get on the route /available_input

        Returned json keys 
        ------------------
        status : str
             status of the request : OK/KO
        comment : str
             comment that can explain status:KO
        prediction: float
             the prediction
        """
        to_predict = request.json

        print(to_predict)
        # is_train = session.get('is_train',False)=="True"
        is_train = data.is_train

        if not is_train :
            app.logger.bug("/predict : not trained yet")
            return jsonify({'status':'KO', 'comment':'not trained yet'}), 401
       	 
        # input_columns = session.get('input_columns',None)
        input_columns = data.input_columns

        pipeline = joblib.load(dumped_pipeline)

        pred = predict(to_predict, input_columns=input_columns, model=pipeline)
        app.logger.info("/predict prediction : %s"%(predict))

        return jsonify({"status":"OK", "prediction":pred})
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')


