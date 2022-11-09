# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelAPIDetails(db.Model):

    __tablename__ = 'api_details'

    api_details_id = Column(Integer, primary_key=True, unique=True)
    api_version = Column(String)
    private_key = Column(String)
    public_key = Column(String)
    request_sample = Column(String)
    model_id = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



