from flask import Blueprint

blueprint = Blueprint(
    'prediction_blueprint',
    __name__,
    url_prefix=''
)