import os 

# define the base path to the emotion dataset
BASE_PATH = "/home/hrushikesh/dl4cv/fer2013/"

# use the base path to define the path to the input emotions file
INPUT_PATH = os.path.sep.join([BASE_PATH, "fer2013.csv"])

# define the number of classes (set to 6, ignoring the "disgust" class)
# NUM_CLASSES = 7
NUM_CLASSES = 6

# define the path to the output training, validation and testing HDF5 files
TRAIN_HDF5 = "/".join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = "/".join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = "/".join([BASE_PATH, "hdf5/test.hdf5"])

# define the batch size
BATCH_SIZE = 128

# define the path to where output logs will be stored 
OUTPUT_PATH = "/".join([BASE_PATH, "output"])

