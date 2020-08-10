# This file is for testing the model, to test the model in own images, use test_predictions.py 

# import necessary packages
from config import gender_age_config as config
from config import age_gender_deploy as deploy
import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
from simplepreprocessor import SimplePreprocessor
from visualize_helper import age_gender_visualize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.keras.models import load_model
import h5py
import numpy as np
import cv2
import argparse
import pickle
from imutils import paths
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample_size", type=int, default=10, help="epoch # to load")
args = vars(ap.parse_args())

# load the label encoders and mean files
print("[INFO] loading label encoder files...")
ageLe = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLe = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())

# load the models from disk
print("[INFO] loading models...")
age_model = load_model(deploy.AGE_MODEL)
gender_model = load_model(deploy.GENDER_MODEL)

# initialize the preprocessor which resizes image
sp = SimplePreprocessor(width=59, height=59, inter=cv2.INTER_CUBIC)

db_age = h5py.File("/home/hrushikesh/images/adience/hdf5/age_test.hdf5","r")
db_gender = h5py.File("/home/hrushikesh/images/adience/hdf5/gender_test.hdf5","r")

# Chose few numbers between 0 to 1693
numbers = np.random.randint(0, 1693, size=[args["sample_size"]])

for i in numbers:
    image = db_age["images"][i]
    age_gender_visualize(image, age_model, gender_model)

    
