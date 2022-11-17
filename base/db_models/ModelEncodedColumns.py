# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_login import UserMixin
from sqlalchemy import BINARY, Column, Integer, String

from app import db


class ModelEncodedColumns(db.Model):
    __tablename__ = 'encoded_columns'

    encoded_columns_id = db.Column(db.Integer, primary_key=True, unique=True)
    column_name = db.Column(db.String)
    column_type = db.Column(db.String)
    model_id = db.Column(db.Integer)
    is_date = db.Column(db.Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)
