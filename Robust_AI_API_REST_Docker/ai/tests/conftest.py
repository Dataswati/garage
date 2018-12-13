import pytest
from flask import Flask
from app import initialize_app


@pytest.fixture
def app():
    app = Flask(__name__)
    initialize_app(app)
    return app
