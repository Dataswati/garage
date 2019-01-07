import logging.config

from flask import Flask, Blueprint
from flask_restplus import Resource, Api

from main_namespace import  ns as main_namespace

# load logging confoguration and create log object
logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)

#create api instance
api = Api(version='1.0',
          title='house pricing prediction REST API',
          description='RESTfull API house pricing prediction')

@api.errorhandler
def default_error_handler(error):
    message = 'Unexpected error occured: {}'.format(error.specific)
    log.exception(message)

# create Flask application
app = Flask(__name__)

log.info("app launch")

def configure_app(flask_app):
    '''
    Configure Flask application

    :param flask_app: instance of Flask() class
    '''
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
#     flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
#     flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
#     flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP
# TODO understand this setup

def initialize_app(flask_app):
    '''
    Initialize Flask application with Flask-RestPlus

    :param flask_app: instance of Flask() class
    '''
    blueprint = Blueprint('API', __name__)

    # configure_app(flask_app)
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
    api.init_app(blueprint)
    
    # init main_namespace
    api.namespaces.clear()
    api.add_namespace(main_namespace)

    flask_app.register_blueprint(blueprint)

initialize_app(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
