# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import request
from flask_login import login_required

from app.base.app_routes.directors.cvision.ObjectDetectionDirector import ObjectDetectionDirector
from app.cvision import blueprint
from bm.core.engine.factories.cvision.ObjectDetectionFactory import ObjectDetectionFactory


## CVision

@blueprint.route('/selectvision', methods=['GET', 'POST'])
@login_required
def selectvision():
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.selectcvisiontype(request)

@blueprint.route('/createmodel/objectdetection', methods=['GET', 'POST'])
@login_required
def createobjectdetection():
    objectdetectionfactory = ObjectDetectionFactory()
    return objectdetectionfactory.createobjectdetectionmodel(request)

