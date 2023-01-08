#!/bin/bash
HOST=$1
USER=$2
PASSWORD=$3
SOURCE=$4
ALL_FILES="${@:5}"

ftp -inv $HOST <<EOF
user $USER $PASSWORD

cd /
cd PycharmProjects/yolov5
cd $SOURCE
rm *
mget $ALL_FILES
bye
EOF