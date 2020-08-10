# import the necessary packages
import os

# define the type of dataset we are training (i.e either "age" or "gender")
DATASET_TYPE = "age"

# define the base paths to the faces dataset and output path
BASE_PATH = "/home/hrushikesh/images/adience"
OUTPUT_BASE = "/home/hrushikesh/dl4cv/gender_age/output"
images = BASE_PATH

# based on the base path, derive the images path and folds path
IMAGES_PATH = os.path.sep.join([BASE_PATH, "aligned"])
LABELS_PATH = os.path.sep.join([BASE_PATH, "folds"])

# define the percentage of validation and testing images relative to the number of training images
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size
BATCH_SIZE = 32
NUM_DEVICES = 1

# check to see if we are working with the "age" portion of the dataset
if DATASET_TYPE == "age":
    # define the number of labels for the "age" dataset, along with the path to the label encoder
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = os.path.sep.join([OUTPUT_BASE, "age_le.cpickle"])

    # define the path to the output training, validation and testing lists
    TRAIN_HDF5 = os.path.sep.join([images, "hdf5/age_train.hdf5"])
    VAL_HDF5 = os.path.sep.join([images, "hdf5/age_val.hdf5"])
    TEST_HDF5 = os.path.sep.join([images, "hdf5/age_test.hdf5"])

    # derive the path to the mean pixel file
    DATASET_MEAN = os.path.sep.join([OUTPUT_BASE, "age_adience_mean.json"])

# otherwise check to see if we are performing "gender" classification

elif DATASET_TYPE == "gender":
    # define the number of labels for the "age" dataset, along with the path to the label encoder
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = os.path.sep.join([OUTPUT_BASE, "gender_le.cpickle"])

    # define the path to the output training, validation and testing lists
    TRAIN_HDF5 = os.path.sep.join([images, "hdf5/gender_train.hdf5"])
    VAL_HDF5 = os.path.sep.join([images, "hdf5/gender_val.hdf5"])
    TEST_HDF5 = os.path.sep.join([images, "hdf5/gender_test.hdf5"])

    # derive the path to the mean pixel file
    DATASET_MEAN = os.path.sep.join([OUTPUT_BASE, "gender_adience_mean.json"])

