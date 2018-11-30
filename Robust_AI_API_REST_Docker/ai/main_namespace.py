import json, requests
import logging
import os

from flask import request
from flask_restplus import Resource, Namespace, reqparse

from rq import Queue
from rq.job import Job
from rq.registry import StartedJobRegistry
from redisQworker import conn

from model import train, predict
import config

train_job_id = "train_job"
log = logging.getLogger(__name__)
q_train = Queue(connection=conn, name='train')
registr_train = StartedJobRegistry('train', connection=conn)


# create dedicated namespace for GAN client
ns = Namespace('/', description='housing prediction')


# Flask-RestPlus specific parser for image uploading
# upload_parser = api.parser()
upload_parser = reqparse.RequestParser()
upload_parser.add_argument("input",
                           type=list,
                           location="json",
                           required=True)


@ns.route('/train')
class Train(Resource):
    @ns.doc(description='Start training on a csv file',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    @ns.expect(upload_parser)
    def post(self):
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
        input_json = request.json
        if not ("target" in input_json.keys()):
            log.bug("/train need a target to predict")
            return {'status':'KO',
                    'comment':'need a target to predict'}, 401
        target = input_json["target"]
        if not ("csv_file_name" in input_json.keys()):
            log.bug("/train need a csv file_name")
            return {'status':'KO',
                    'comment':'need a csv file_name'}, 401
        csv_file_name = input_json["csv_file_name"]
        csv_path = os.path.join(config.folder_csv_path, csv_file_name)

        if not ("input_columns" in input_json.keys()):
            input_columns = None
        else :
            input_columns = input_json["input_columns"]

        if not ("restart_train" in input_json.keys()):
            restart_train=False
        else :
            restart_train = input_json["input_columns"]

        job = q_train.fetch_job(train_job_id)
        if not (job is None):
            if job.get_status() in ['started','queued']:
                job_finish=False
            else:
                job_finish=True
        else:
            job_finish=True
        if (not restart_train) and (not job_finish):
            return {'status':'ko',
                    'comment':'previous training not finished'},401
        elif not job_finish:
            job.delete()

        log.info("/train start the train")
#         best_test_score, best_pipeline, input_columns = train(csv_path,
#                         target=target,input_columns=input_columns)
        job_train = q_train.enqueue(train,job_id=train_job_id,args=(csv_path,),
                kwargs={'target':target,'input_columns':input_columns},result_ttl=-1)
        # session["input_columns"] = input_columns
        # data.input_columns = input_columns
        # joblib.dump(best_pipeline, dumped_pipeline)
        # log.info("/train model dumped")
    
        # session["is_train"]="True"
        # data.is_train = True

        # kwargs = json_decode["content"]
        # id = json_decode["id"]
        log.info("input : ")
        log.info(repr(input_json))
        # start now
        # job_diag = q_diag.enqueue(to_execute,kwargs)
        # job_diag_id = job_diag.get_id()

        return {'status':'ok'}, 200
    def get(self):

        job = q_train.fetch_job(train_job_id)
        if job is None :
            return{'status':"ok","train_status":"not_trained"}, 200
        else :
            return {'status':"ok",
                    'train_status':job.get_status(),
                    'train_exec_info':job.exc_info}, 200

    def delete(self):

        job = q_train.fetch_job(train_job_id)
        if not (job is None):
            job.delete()
            return {'status':'ok'},200
        else :
            return {'status':'ko','comment':'no train to delete'},401


@ns.route('/predict')
class Predict(Resource):
    @ns.doc(description='Predict from json input',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    @ns.expect(upload_parser)
    def post(self):
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
        log.info("predict input : \n %s"%(repr(to_predict),))
        job = q_train.fetch_job(train_job_id)
        if not (job is None):
            if job.get_status() in ['started','queued']:
                job_finish=False
            else:
                job_finish=True
        else:
            job_finish=True

        if not job_finish :
            log.bug("/predict : not trained yet")
            return jsonify({'status':'ko', 'comment':'not trained yet'}), 401

        best_test_score, pipeline, input_columns= job.result
        pred = predict(to_predict, input_columns=input_columns,
                model=pipeline)
        log.info("/predict prediction : %s"%(predict,))

        return {"status":"ok", "prediction":pred},200
        # return prediction
    def get(self):
        job = q_train.fetch_job(train_job_id)
        if not (job is None):
            if job.get_status() in ['started','queued']:
                job_finish=False
            else:
                job_finish=True
        else:
            job_finish=True

        if not job_finish :
            log.bug("/predict : not trained yet")
            return jsonify({'status':'ko', 'comment':'not trained yet'}), 401

        best_test_score, pipeline, input_columns= job.result

        return jsonify({'status':'OK','input_names':input_columns})
        # return json sample

