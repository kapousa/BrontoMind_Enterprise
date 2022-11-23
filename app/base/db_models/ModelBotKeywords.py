# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from sqlalchemy import Column, Integer, String

from app import db


class ModelBotKeywords(db.Model):

    __tablename__ = 'bot_keywords'

    id = Column(Integer, primary_key=True, unique=True)
    keywords = Column(String)
    model_type = Column(String)
    model_code = Column(Integer)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, property, value)

    def __repr__(self):
        return str(self.model_id)



