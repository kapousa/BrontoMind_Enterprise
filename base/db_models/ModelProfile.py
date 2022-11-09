# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelProfile(db.Model):

    __tablename__ = 'model_profile'

    model_id = Column(Integer, primary_key=True, unique=True)
    model_name = Column(String)
    user_id = Column(Integer)
    model_headers = Column(String)
    prediction_results_accuracy = Column(String)
    mean_absolute_error = Column(String)
    mean_squared_error = Column(String)
    root_mean_squared_error = Column(String)
    mean_percentage_error = Column(String)
    mean_absolute_percentage_error = Column(String)
    plot_image_path = Column(String)
    created_on = Column(String)
    updated_on = Column(String)
    last_run_time = Column(String)
    ds_source = Column(String)
    ds_goal = Column(String)
    mean_percentage_error = Column(String)
    mean_absolute_percentage_error = Column(String)
    depended_factor = Column(String)
    forecasting_category = Column(String)
    train_precision = Column(String)
    train_recall = Column(String)
    train_f1 = Column(String)
    test_precision = Column(String)
    test_recall = Column(String)
    test_f1 = Column(String)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



