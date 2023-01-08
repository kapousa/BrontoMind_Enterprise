#!/bin/bash
cd /Users/kapo/PycharmProjects/yolov5/runs/detect
cd $1
zip -r $1 ./*
mv $1.zip /Users/kapo/PycharmProjects/BrontoMind_Enterprise/app/detection
rm $1.zip
rm -rf /Users/kapo/PycharmProjects/yolov5/runs/detect/*