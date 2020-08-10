# Used to test the model on images. Usage python test_predictions --image path_to_image/directory_containing_images

import cv2
# import the necessary packages
import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
from config import gender_age_config as config
from config import age_gender_deploy as deploy
from simplepreprocessor import SimplePreprocessor
from visualize_helper import age_gender_visualize
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import json
import dlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.keras.models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image (or directory)")
args = vars(ap.parse_args())

# load the label encoders 
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())

# load the models from disk
print("[INFO] loading models...")
age_model = load_model(deploy.AGE_MODEL)
gender_model = load_model(deploy.GENDER_MODEL)

# initialize the preprocessor which resizes image
sp = SimplePreprocessor(width=59, height=59, inter=cv2.INTER_CUBIC)

# initialize dlib’s face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
imagePaths = [args["image"]]

# if the input path is actually a directory, then list all image paths in the directory
if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))

# loop over the image paths
for imagePath in imagePaths:
    # load the image from disk, resize it and convert it to grayscale
    print("[INFO] processing {}:".format(imagePath))
    image = cv2.imread(imagePath)
    #image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detection
    for rect in rects:
        # determine the facial landmarks for the face region, then align the face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # resize the face to a fixed size, then extract 10-crop patches from it
        face = sp.preprocess(face)
        age_gender_visualize(face, age_model, gender_model)         


