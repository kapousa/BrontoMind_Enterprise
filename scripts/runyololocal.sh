#! /bin/bash
cd /Users/kapo/PycharmProjects/yolov5
source venv/bin/activate
python3 detect.py --source $1 --name $2
cd $1
find . -maxdepth 1 -type d -exec zip archive.zip {} +
ls
rm -rf /Users/kapo/PycharmProjects/yolov5/data/images/*
rm -rf /Users/kapo/PycharmProjects/yolov5/data/videos/*