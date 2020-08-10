# import the necessary packages
import sys
sys.path.append("/home/hrushikesh/dl4cv/preprocessing")
sys.path.append("/home/hrushikesh/dl4cv/gender_age/config")
sys.path.append("/home/hrushikesh/dl4cv/io")
import gender_age_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from agegenderhelper import AgeGenderHelper
from aspectawarepreprocessor import AspectAwarePreprocessor
from hdf5datasetwriter import HDF5DatasetWriter
import numpy as np
import progressbar
import pickle
import json
import cv2

# initialize our helper class, then build the set of image paths and class labels
print("[INFO] building paths and labels...")
agh = AgeGenderHelper(config)
(trainPaths, trainLabels) = agh.buildPathsAndLabels()

# now that we have the total number of images in the dataset that can be used for training, compute the number of images for validation and testing
numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

# our class labels are represented as strings so we need to encode them
print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

# perform sampling from the training set to construct a validation set
print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal, stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# perform stratified sampling from training set to construct a testing set
print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest, stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# construct a list pairing the training, validation and testing image paths along with their corresponding labels and output list files
datasets = [
        ("train", trainPaths, trainLabels, config.TRAIN_HDF5), 
        ("val", valPaths, valLabels, config.VAL_HDF5),
        ("test", testPaths, testLabels, config.TEST_HDF5)]

# initialize the list of RGB channel averages
(R, G, B) = ([], [], [])

aap = AspectAwarePreprocessor(59, 59)
# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 59, 59, 3), outputPath, bufSize=500)

    # initialize the progress bar
    widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over each of the individual images + labels
    for (i, (image_path, label)) in enumerate(zip(paths, labels)):
        
        # counter to give how many images missed
        counter = 0
        image = cv2.imread(image_path)
        if image is None:
            counter += 1
            continue

        image = np.uint8(aap.preprocess(image))
        writer.add([image], [label])
        pbar.update(i)
        
        # if we are building the training dataset, then compute the mean of each channel in the image, then update the respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
    print("Counter:{}/{}".format(counter, len(paths)))
    writer.close()
    # close the output file
    pbar.finish()


# construct a dictionary of averages, then serialize the means to a JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

# serialize the label encoder
print("[INFO] serializing label encoder...")
f = open(config.LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
