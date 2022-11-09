# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelAPIModelMethods(db.Model):
    __tablename__ = 'api_model_methods'

    id = Column(Integer, primary_key=True, unique=True)
    method_name = Column(String)
    method_description = Column(String)
    url = Column(String)
    sample_request = Column(String)
    sample_response = Column(String)
    api_details_id = Column(Integer)
    model_goal = Column(Integer)
    model_id = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.id)
