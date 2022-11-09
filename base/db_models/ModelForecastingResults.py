# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelForecastingResults(db.Model):

    __tablename__ = 'forecasting_results'

    forecasting_results_id = Column(Integer, primary_key=True, unique=True)
    actual = Column(String)
    predicted = Column(String)
    period_dates = Column(String)
    model_id = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



