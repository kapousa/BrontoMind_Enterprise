#  Copyright (c) 2021. Slonos Labs. All rights Reserved.

import mysql.connector


class DBConnector:

    def __init__(self):
        self.host = ''
        self.user = ''
        self.password = ''
        self.db_name = ''

    def create_mysql_connection(self, host, user, password, db_name):
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db_name
        )
        return mydb
