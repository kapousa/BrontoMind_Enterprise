# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelCvisionRun(db.Model):

    __tablename__ = 'cvision_run'

    id = Column(Integer, primary_key=True, unique=True)
    run_id = Column(Integer)
    model_id = Column(Integer)
    run_on = Column(String)
    description = Column(String)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



