import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
sys.path.append("/home/hrushikesh/dl4cv/io")
sys.path.append("/home/hrushikesh/dl4cv/gender_age/config")
import gender_age_config as config
from oneOff import buildOneOffMappings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pickle
import numpy as np
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to model checkpoint to load")
ap.add_argument("-t", "--type", type=str, default="age", help="the dataset type we are dealing gender or age")
args = vars(ap.parse_args())

# Reduce the batch size to 1, since one-off accuracy has only been tested for batch size = 1
config.BATCH_SIZE = 1

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, classes=config.NUM_CLASSES)

# load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# evaluate the network
(loss, acc) = model.evaluate_generator(
        testGen.generator(),
        steps = testGen.numImages // config.BATCH_SIZE,
        max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] rank-1 accuracy: {:.2f}".format(acc * 100))


# If the dataset_type is age, calculate the oneOff accuracy as well
if args["type"] == "age":
    # load label encoder file
    le = pickle.loads(open("/home/hrushikesh/dl4cv/gender_age/output/age_le.cpickle", "rb").read())
    
    # Create the dictionary of oneOffLabels which contains the classes adjacent to current class
    oneOffLabels = buildOneOffMappings(le)
    
    # Initialize few helper variables to keep a count of for oneOff accuracy 
    counter = 0
    total = 0
    for (image, labels) in testGen.generator(passes=1):
        predictions = model.predict(image)
        predictions = predictions.flatten()
        ind = predictions.argmax()
        label = int(np.where(labels == 1)[1])
        
        if ind in oneOffLabels[label]:
            counter += 1
        
        total += 1

    accuracy = counter * 100 / total
    print("One-Off accuracy: {}".format(round(accuracy,2)))

# close the testing database
testGen.close()

