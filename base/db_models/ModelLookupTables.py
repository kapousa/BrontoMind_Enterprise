# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelLookupTable(db.Model):

    __tablename__ = 'lookup_table'

    key = Column(Integer, primary_key=True, unique=True)
    value = Column(String)
    cat_id = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.cat_id)



