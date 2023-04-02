#!/bin/bash
cd /mnt/ssd1/yuecong/data/ARID
for fullfile in /mnt/ssd1/yuecong/data/ARID/clips_v1/Drink/*.mp4; do
    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"
    mkdir $filename
    cd $filename
    ffmpeg -i $fullfile -threads 1 -vf scale=-1:256 -q:v 0 "%06d.jpg"
    cd ..
    break
done