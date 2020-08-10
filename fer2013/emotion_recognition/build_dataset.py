from config import emotion_config as config
import sys
sys.path.append("/home/hrushikesh/dl4cv/io")
from hdf5datasetwriter import HDF5DatasetWriter
import numpy as np
import cv2

# open the file for reading (skipping the header), then initialize the list of data and labels for the training, validation and testing sets
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()
#>>> f.__next__()
#'emotion,pixels,Usage\n'

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# loop over the rows in the input file 
for row in f:
    # extract the label, image and usage from the row
    (label, image, usage) = row.strip().split(",")
    label = int(label)

    # if we are ignoring the "disgust" class there will be 6 total 
    # class labels instead of 7
    if config.NUM_CLASSES == 6:
        # merge the anger and disgust classes
        if label == 1:
            label = 0

        # if label has a value greated than zero, subtract one from it to
        # make all labels sequential (not required but helps in interpreting results
        if label > 0:
            label -= 1

    # reshape the flattened pizel list into a 48 * 48 grayscale image
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48,48))
    
    # check if we are examining a training or testing or validation image
    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    else:
        testImages.append(image)
        testLabels.append(label)
# construct a list pairing the training, validation and testing images along with their corresponding labels and output HDF5 files
print(len(trainImages), len(testImages), len(valImages))
print(trainImages[2].shape, testImages[2].shape, valImages[2].shape)
datasets = [
        (trainImages, trainLabels, config.TRAIN_HDF5),
        (valImages, valLabels, config.VAL_HDF5),
        (testImages, testLabels, config.TEST_HDF5)]

f.close()
# loop over the dataset tuples
for (images, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}".format(outputPath))
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)
    
    # loop over the image and add them ti the dataset
    for (image, label) in zip(images, labels):
        #print(image.shape)
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        writer.add([image], [label])

    # close the HDF5 writer
    writer.close()


