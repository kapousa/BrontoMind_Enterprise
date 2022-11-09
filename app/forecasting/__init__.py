from flask import Blueprint

blueprint = Blueprint(
    'forecasting_blueprint',
    __name__,
    url_prefix=''
)