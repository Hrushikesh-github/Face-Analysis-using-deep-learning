# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
sys.path.append("/home/hrushikesh/dl4cv/conv")
sys.path.append("/home/hrushikesh/dl4cv/callbacks")
sys.path.append("/home/hrushikesh/dl4cv/io")
sys.path.append("/home/hrushikesh/dl4cv/gender_age/config")
import gender_age_config as config
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
from hdf5datasetgenerator import HDF5DatasetGenerator
#from minigooglenet import MiniGoogLeNet
from my_model import My_Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", default=None, help="path to trained-model")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# Initialize data augmentation with only horizontal_flip and rotation_range upto 7 degrees only 
aug=ImageDataGenerator(rotation_range=7,horizontal_flip=True, fill_mode="nearest")

# Initialize the generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug, classes=8)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32, classes=8)

model = My_Model.build(width=59, height=59, depth=3, classes=8)
if args["model"] is None:  
  print("[INFO] compiling model...")
  opt = SGD(lr=1e-3, momentum=0.9)
  model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
  print("[INFO] loading {}...".format(args["model"]))
  model = load_model(args["model"])
  # update the learning rate
  print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))  
  K.set_value(model.optimizer.lr, 1e-5)  
  print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [
             EpochCheckpoint(args["checkpoints"], every=5,startAt=args["start_epoch"]),
             TrainingMonitor("output/result.png", jsonPath="output/result.json", startAt=args["start_epoch"])]

print("[INFO] training head....")

# train the network
model.fit_generator(
        trainGen.generator(),
        validation_data = valGen.generator(),
        epochs=65,
        steps_per_epoch=trainGen.numImages // 32,
        validation_steps=valGen.numImages // 32,
        callbacks=callbacks, 
        verbose=1,
        initial_epoch=args["start_epoch"])

