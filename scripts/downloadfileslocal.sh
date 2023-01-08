#!/bin/bash
SOURCE=$1
ALL_FILES="${@:2}"
cd /
cd PycharmProjects/yolov5
cd $SOURCE
rm *
mget $ALL_FILES
bye
EOF