# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template
from flask_login import login_required

from app.cvision import blueprint

## CVision

@blueprint.route('/')
@login_required
def index():
    return render_template('page-404.html')
