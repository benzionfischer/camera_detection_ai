#!/bin/bash


# step 1
conda echo 'export PATH="/opt/homebrew/anaconda3/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
conda env list
conda create --name cctv_detection python=3.8.19
conda activate cctv_detection

# step 2
# create a new project in intellij
# set conda as default env with the detials from 'step 1'


# venv & dependencies
python3.8 -m venv venv
pip install ultralytics
pip install tensorflow
pip install tensorflow-macos
pip install tensorflow-metal

# for each problematic .png file use
mogrify *.png


# for raspberry pi (from scrach)
python -m venv venv
pip install opencv-python
pip install ultralytics
