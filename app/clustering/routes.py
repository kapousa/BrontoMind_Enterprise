# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from flask import render_template, request
from flask_login import login_required

from app.base.app_routes.directors.ClusteringDirector import ClusteringDirector
from app.clustering import blueprint

## Clustering
from bm.controllers.BaseController import BaseController


@blueprint.route('/downloadlabledfile', methods=['GET', 'POST'])
@login_required
def downloadlabledfile():
    return ClusteringDirector.download_labeled_datafile()


@blueprint.route('/getdatacluster', methods=['GET', 'POST'])
def getdatacluster():
    try:
        model_status = BaseController.get_model_status()

        if (len(model_status) == 0):
            return render_template('applications/pages/nomodeltopredictevalues.html',
                                   message='There is no active model')

        ds_goal = request.args.get("t")
        clustering_director = ClusteringDirector()

        return clustering_director.get_clusters(request)
    except Exception as e:
        return render_template('applications/pages/nomodeltopredictevalues.html',
                               error=str(e),
                               message="Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset",
                               segment='message')