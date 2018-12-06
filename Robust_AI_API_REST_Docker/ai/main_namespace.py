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
import datetime

id_time_format = "%Y-%m-%d %H:%M:%S"
train_job_id = "train_job"
train_new_job_id = "new_train_job"
log = logging.getLogger(__name__)
q_train = Queue(connection=conn, name='train')
registr_train = StartedJobRegistry('train', connection=conn)

def init_time_id(id_time_format=id_time_format):
    return datetime.datetime.utcnow().strftime(id_time_format)

def sort_id_time(list_id_time, id_time_format=id_time_format):
    list_datetime = [datetime.datetime.strptime(x,id_time_format)
            for x in list_id_time]
    list_datetime = sorted(list_datetime)
    sorted_list = [x.strftime(id_time_format) for x in list_datetime]
    return sorted_list


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
            restart_train = input_json["restart_train"]

        job = q_train.fetch_job(train_job_id)
        job_new = q_train.fetch_job(train_new_job_id)

        # 4 case posible :
        # * start job
        # * start new job
        # * move new job to job and start new
        # * delete new and start new
        def job_trained(job):
            if job.get_status() in ['started','queued']:
                return False
            else:
                return True 

        if job is None :
            # case start job
            job_train = q_train.enqueue(train,
                    job_id=train_job_id,args=(csv_path,),
                kwargs={'target':target,'input_columns':input_columns},result_ttl=-1)
        elif job_new is None:
            # case start new job
            job_train = q_train.enqueue(train,
                    job_id=train_new_job_id,args=(csv_path,),
                    kwargs={'target':target,
                            'input_columns':input_columns},
                    result_ttl=-1)
        elif job_trained(job_new):
            #case move new job to job and start new
            job.delete()
            job_new.set_id(train_job_id)
            job_train = q_train.enqueue(train,
                    job_id=train_new_job_id, args=(csv_path,),
                    kwargs={'target':target,
                            'input_columns':input_columns},
                    result_ttl=-1)
        elif restart_train :
            # case delete new and start new
            job_new.delete()
            job_train = q_train.enqueue(train,job_id=train_new_job_id,args=(csv_path,),
                kwargs={'target':target,'input_columns':input_columns},result_ttl=-1)
        else :
            return {'status':'ko',
                    'comment':'previous training not finished'},401


        # log.info("/train start the train")
        log.info("input : ")
        log.info(repr(input_json))
        return {'status':'ok'}, 200
    def get(self):

        job = q_train.fetch_job(train_job_id)
        job_new = q_train.fetch_job(train_new_job_id)
        print("job ",job)
        print("job_new ",job_new)
        if job is None :
            return{'status':"ok",
                    "training":False,
                   "model_available":False}, 200
        elif job.get_status() in ['started','queued']:
            return {'status':"ok",
                    'training':True,
                    'model_available':False}, 200
        elif job_new is None :
            return {'status':"ok",
                    'training':False,
                    'model_available':True}, 200
        elif job_new.get_status() in ['started','queued']:
            return {'status':"ok",
                    'training':True,
                    'model_available':True}, 200
        else :
            return {'status':"ok",
                    'training':False,
                    'model_available':True}, 200





    def delete(self):

        job = q_train.fetch_job(train_job_id)
        job_new = q_train.fetch_job(train_new_job_id)
        to_return = {"status":'ok',
                'number_delete_model':sum([not (job is None),
                    not (job_new is None)])}
        if not (job is None):
            job.delete()
        if not (job_new is None):
            job_new.delete()
        
        return to_return, 200

def get_job():
    job = q_train.fetch_job(train_job_id)
    job_new = q_train.fetch_job(train_new_job_id)
    if not (job_new is None):
        if not (job_new.get_status() in ['started','queued']):
            job_finish=True
            job.delete()
            job_new.set_id(train_job_id)
            job = job_new
        elif job.get_status() in ['started','queued']:
            job_finish=False
        else :
            job_finish=True



    elif not (job is None):
        if job.get_status() in ['started','queued']:
            job_finish=False
        else:
            job_finish=True
    else:
        job_finish=False
    return job, job_finish

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
        job, job_finish = get_job()

        if not job_finish :
            log.bug("/predict : not trained yet")
            return {'status':'ko', 'comment':'not trained yet'}, 401

        best_test_score, pipeline, input_columns= job.result
        pred = predict(to_predict, input_columns=input_columns,
                model=pipeline)
        log.info("/predict prediction : %s"%(predict,))

        return {"status":"ok", "prediction":pred},200
        # return prediction
    def get(self):
        
        job, job_finish = get_job()
        if not job_finish :
            log.bug("/predict : not trained yet")
            return jsonify({'status':'ko', 'comment':'not trained yet'}), 401

        best_test_score, pipeline, input_columns= job.result

        return {'status':'ok','input_names':input_columns},200

