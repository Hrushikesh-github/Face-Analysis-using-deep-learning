import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
sys.path.append("/home/hrushikesh/dl4cv/callbacks")
sys.path.append("/home/hrushikesh/dl4cv/io")
sys.path.append("/home/hrushikesh/dl4cv/fer2013")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from config import emotion_config as config
from emotionvggnet import EmotionVGGNet
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
from sklearn.metrics import classification_report

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing image generators for data augmentation,then initialize the image preprocessors
trainAug = ImageDataGenerator(horizontal_flip=True, rescale=1/ 225.0)#, zoom_range=0.1, rotation_range=10, rescale=1/ 255.0) 
valAug = ImageDataGenerator(rescale=1/ 255.0)
iap = ImageToArrayPreprocessor()

# initalize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initalize the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    #opt = Adam(lr=1e-3)
    opt = SGD(lr=1e-2, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# otherwise load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate:{}".format(K.get_value(model.optimizer.lr)))
    
# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
        TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"])]
# train the network
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch = trainGen.numImages // config.BATCH_SIZE,
        validation_data = valGen.generator(),
        validation_steps = valGen.numImages // config.BATCH_SIZE,
        epochs = 70,
        max_queue_size = config.BATCH_SIZE * 2,
        initial_epoch = args["start_epoch"],
        callbacks = callbacks, verbose = 2)

# close the databases
trainGen.close()
valGen.close()

