#!/bin/bash

cp -r /mnt/dihu/ffmpeg-7.0.2-amd64-static/ ./ffmpeg
export PATH="./ffmpeg:$PATH"
echo $PATH
ffmpeg -version