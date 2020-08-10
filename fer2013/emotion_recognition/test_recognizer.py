import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
sys.path.append("/home/hrushikesh/dl4cv/io")
sys.path.append("/home/hrushikesh/dl4cv/fer2013")
from config import emotion_config as config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize the testing data generator and image preprocessor
testAug =  ImageDataGenerator(rescale=1/ 255.0)
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug=testAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# evaluate the network
(loss, acc) = model.evaluate_generator(
        testGen.generator(),
        steps = testGen.numImages // config.BATCH_SIZE,
        max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
testGen.close()
