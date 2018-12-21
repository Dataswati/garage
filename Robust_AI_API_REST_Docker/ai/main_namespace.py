import json, requests
import logging
import os
from collections import OrderedDict

from flask import request
from flask_restplus import Resource, Namespace, reqparse, fields

from rq import Queue
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry
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

started_registry = StartedJobRegistry('train',connection=conn)
finished_registry = FinishedJobRegistry('train',connection=conn)

def init_time_id(id_time_format=id_time_format):
    return datetime.datetime.utcnow().strftime(id_time_format)

def sort_id_time(list_id_time, id_time_format=id_time_format, reverse=False):
    list_datetime = [datetime.datetime.strptime(x,id_time_format)
            for x in list_id_time]
    list_datetime = sorted(list_datetime, reverse=reverse)
    sorted_list = [x.strftime(id_time_format) for x in list_datetime]
    return sorted_list


ns = Namespace('/', description='')

input_post_train = ns.model('post train input', OrderedDict(
        target=fields.String(
            description="The target to predict present as a column in the csv", required=True),
        csv_file_name=fields.String(
            description="The name of the csv file present in de csv folder use for the training", required=True),
        input_columns=fields.String(
            description="The list of the columns that should be used as the input, if not specified", required=False),
        restart_train=fields.Boolean(
            description="if True delete the current training model to retrain on the new parameters", required=False)
        ))
#             all the columns execpt the target will be used
output_error = ns.model('error output', {
        'status': fields.String(
            description="status of the request : OK/KO"),
        'comment': fields.String(
            description="comment on KO")
        })
output_standard = ns.model('standard output', {
        'status': fields.String(
            description="status of the request : OK/KO")
        })
output_get_train = ns.model('get train output',{
        'status': fields.String(
            description="status of the request : OK/KO"),
        'model_available':fields.Boolean(
            description="if True model is ready for prediction"),
        'started_train':fields.List(fields.DateTime(),
            description="list of datetime of started train on model"),
        'finished_train':fields.List(fields.DateTime(),
            description="list of datetime of finished train on model"),
        'queued_train':fields.List(fields.DateTime(),
            description="list of datetime of queued train on model")
    })
#         return {'status':"ok",
#                 'training' : training,
#                 'model_available' : model_available,
#                 'started_train' : started_registry.get_job_ids(),
#                 'finished_train' : finished_registry.get_job_ids(),
#                 'queued_train' : q_train.get_job_ids()}, 200

@ns.route('/train')
class Train(Resource):

    @ns.doc(description='launch the train on the specified csv file with the specified target',
            body=input_post_train)
    @ns.response(200,"success",output_standard)
    @ns.response(401,"bad request",output_error)
    @ns.response(500,"internal server error")
    def post(self):
        input_json = request.json
        if not ("target" in input_json.keys()):
            log.debug("/train need a target to predict")
            return {'status':'KO',
                    'comment':'need a target to predict'}, 401
        target = input_json["target"]
        if not ("csv_file_name" in input_json.keys()):
            log.debug("/train need a csv file_name")
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

        job_train = q_train.enqueue(train,
                    job_id=init_time_id(),args=(csv_path,),
                kwargs={'target':target,'input_columns':input_columns},result_ttl=-1)
        return {'status':'ok'}, 200
    @ns.doc(description='display the different queuing training or finished model by their timestamp id')
    @ns.response(200,"success",output_get_train)
    @ns.response(401,"bad request",output_error)
    @ns.response(500,"internal server error")
    def get(self):
        job_ids = q_train.get_job_ids() +\
                started_registry.get_job_ids()+\
                finished_registry.get_job_ids()
        job_ids = list(set(job_ids))
        #print(job_ids)
        sorted_job_ids = sort_id_time(job_ids, reverse=True)
        training = False
        model_available = False
        for job_id in sorted_job_ids :
            job = q_train.fetch_job(job_id)
            if job.get_status() in ['started','queued']:
                training=True
            else :
                model_available=True
                break

        return {'status':"ok",
                'training' : training,
                'model_available' : model_available,
                'started_train' : started_registry.get_job_ids(),
                'finished_train' : finished_registry.get_job_ids(),
                'queued_train' : q_train.get_job_ids()}, 200


    @ns.doc(description='remove all models of the system')
    @ns.response(200,"success",output_standard)
    @ns.response(500,"internal server error")
    def delete(self):

        job_ids = q_train.get_job_ids() +\
                started_registry.get_job_ids()+\
                finished_registry.get_job_ids()
        job_ids = list(set(job_ids))
        for job_id in job_ids:
            job = q_train.fetch_job(job_id)
            job.delete()
        
        return {'status' : 'ok'}, 200

def get_job():
    job_ids = q_train.get_job_ids() +\
            started_registry.get_job_ids()+\
            finished_registry.get_job_ids()
    job_ids = list(set(job_ids))
    sorted_job_ids = sort_id_time(job_ids, reverse=True)
    model_available = False
    job = None
    for job_id in sorted_job_ids :
        job = q_train.fetch_job(job_id)
        if model_available:
            job.delete()
        else:
            if not (job.get_status() in ['started','queued']):
                model_available=True

    return job, model_available

input_post_predict = ns.model('post predict input', {
    "key_1":fields.Float(
            description="input value corresponding to key_1", required=True),
    "key_2":fields.Float(
            description="input value corresponding to key_2", required=True)
        },description="key:value were key are the input_columns keys that are returned by method get on route predict")

output_post_predict = ns.model('post predict output', {
        'status': fields.String(
            description="status of the request : OK/KO"),
        'prediction' : fields.Float(descripition='the prediction value')
        })
output_get_predict = ns.model('post predict output', {
        'status': fields.String(
            description="status of the request : OK/KO"),
        'input_names' : fields.List(fields.String(descripition='the input keys'))
        })

@ns.route('/predict')
class Predict(Resource):

    @ns.doc(description='Predict from json input',
            body=input_post_predict)
    @ns.response(200,"success",output_post_predict)
    @ns.response(401,"bad request",output_error)
    @ns.response(500,"internal server error")
    def post(self):
        to_predict = request.json
        log.info("predict input : \n %s"%(repr(to_predict),))
        job, job_finish = get_job()

        if not job_finish :
            log.debug("/predict : not trained yet")
            return {'status':'ko', 'comment':'not trained yet'}, 401

        best_test_score, pipeline, input_columns= job.result
        if set(to_predict.keys())!= set(input_columns):
            return {'status':'ko', 'comment':'input keys do not match model keys'}, 400
        pred = predict(to_predict, input_columns=input_columns,
                model=pipeline)
        log.info("/predict prediction : %s"%(predict,))

        return {"status":"ok", "prediction":pred},200


    @ns.doc(description='Return the input key of the model')
    @ns.response(200,"success",output_get_predict)
    @ns.response(401,"bad request",output_error)
    @ns.response(500,"internal server error")
    def get(self):
        
        job, job_finish = get_job()
        if not job_finish :
            log.debug("/predict : not trained yet")
            return {'status':'ko', 'comment':'not trained yet'}, 401

        best_test_score, pipeline, input_columns= job.result

        return {'status':'ok','input_names':input_columns},200

