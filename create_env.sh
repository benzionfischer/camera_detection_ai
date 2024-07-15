#!/bin/bash


# step 1
conda echo 'export PATH="/opt/homebrew/anaconda3/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
conda env list
conda create --name ben_env python=3.8.19
conda activate ben_env

# step 2
# create a new project in intellij
# set conda as default env with the detials from 'step 1'
