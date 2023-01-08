#! /bin/bash
cd /Users/kapo/PycharmProjects/yolov5
source venv/bin/activate
python3 detect.py --source $1 --name $2
echo $1
echo $2