# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelAPIMethods(db.Model):
    __tablename__ = 'api_methods'

    api_method_id = Column(Integer, primary_key=True, unique=True)
    method_name = Column(String)
    method_description = Column(String)
    url = Column(String)
    sample_request = Column(String)
    sample_response = Column(String)
    model_goal = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.api_method_id)
