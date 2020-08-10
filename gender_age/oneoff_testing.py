# This file was used to experiment, generate and test oneoff accuracy. It works for batch size 1 and I have not implemented it on other batch sizes
# This file can be ignored

import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
sys.path.append("/home/hrushikesh/dl4cv/io")
sys.path.append("/home/hrushikesh/dl4cv/gender_age/config")
import pickle
import gender_age_config as config
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import argparse


# initialize the testing data generator and image preprocessor
# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 1, classes=config.NUM_CLASSES)
# load the model from disk
args = {"model": "/home/hrushikesh/dl4cv/gender_age/checkpoints/age/epoch_55.hdf5"}
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

def buildOneOffMappings(le):
    # sort the class labels in ascending order (according to age) and initialize the one off mappings for computing accuracy
    classes = sorted(le.classes_, key=lambda x: int(x.split("_")[0]))
    oneOff = {}

    # loop over the index and name of the (sorted) class labels
    for (i, name) in enumerate(classes):
        # determine the index of the *current* class label name
        # in the *label encoder* (unordered) list, then
        # initialize the index of the previous and next age
        # groups adjacent to the current label
        name = str(name)
        current = np.where(le.classes_ == name)[0][0]
        prev = -1
        Next = -1

        # check to see if we should compute previous adjacent
        # age group
        if i > 0:
            prev = np.where(le.classes_ == classes[i - 1])[0][0]
        # check to see if we should compute the next adjacent
        # age group
        if i < len(classes) - 1:
            Next = np.where(le.classes_ == classes[i + 1])[0][0]

        # construct a tuple that consists of the current age
        # bracket, the previous age bracket, and the next age
        # bracket
        oneOff[current] = (current, prev, Next)

    # return the one-off mappings
    return oneOff


le = pickle.loads(open("/home/hrushikesh/dl4cv/gender_age/output/age/age_le.cpickle", "rb").read())
oneOffLabels = buildOneOffMappings(le)

counter = 0
total = 0
for (image, labels) in testGen.generator(passes=1):
    print(image.shape)
    predictions = model.predict(image)
    predictions = predictions.flatten()
    print("\npredictions: ",predictions)
    print("\nlabels: ",labels)
    #ind = list(np.argpartition(predictions, -1, axis=0)[-1:])
    ind = predictions.argmax()
    label = int(np.where(labels == 1)[1])
    
    print("\nindex: ",ind)
    print("\nlabel: ",label)
    print("\noneOffLabel value: ",oneOffLabels[label])
    if ind in oneOffLabels[label]:
        #print("HERE", i)
        counter += 1
    

    total += 1
    time.sleep(6)

accuracy = counter * 100 / total
print("total images: ", total)
print("acc", accuracy)
print(accuracy)

