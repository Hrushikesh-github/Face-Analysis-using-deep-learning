import mxnet as mx
import argparse
import logging
import pickle
import json
import numpy as np
import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

class MxAlexNet:
    @staticmethod
    def build(classes):

        # data input
        data = mx.sym.Variable("data")
        # Block #1: first CONV => relu => POOL layer set
        conv1_1 = mx.sym.Convolution(data=data, kernel=(5, 5), pad=(2,2), num_filter=32) # (59, 59, 32)
        act1_1 = mx.sym.Activation(data=conv1_1, act_type="relu")
        bn1_1 = mx.sym.BatchNorm(data=act1_1)
        conv1_2 = mx.sym.Convolution(data=bn1_1, kernel=(5, 5), pad=(2,2), num_filter=64) # (59, 59, 64)
        do1 = mx.sym.Dropout(data=conv1_2, p=0.25)

        # Block #2: second CONV => relu => POOL layer set
        conv2_1 = mx.sym.Convolution(data=do1, kernel=(3, 3), pad=(1, 1), num_filter=64) # (59, 59, 64)
        act2_1 = mx.sym.Activation(data=conv2_1, act_type="relu")
        bn2_1 = mx.sym.BatchNorm(data=act2_1)
        conv2_2 = mx.sym.Convolution(data=bn2_1, kernel=(3, 3), pad=(1, 1), num_filter=128) # (59, 59, 128)
        do2 = mx.sym.Dropout(data=conv2_2, p=0.25)

        # Block #3: second CONV => relu => POOL layer set
        conv3_1 = mx.sym.Convolution(data=do2, kernel=(3, 3), stride=(2,2), num_filter=128) # 29 29 128
        act3_1 = mx.sym.Activation(data=conv3_1, act_type="relu")
        bn3_1 = mx.sym.BatchNorm(data=act3_1)
        conv3_2 = mx.sym.Convolution(data=bn3_1, kernel=(3, 3), num_filter=128) # 27 27 128

        #pool3 = mx.sym.Pooling(data=bn3_1, pool_type="max", kernel=(3,3)) # 27 27 128
        do3 = mx.sym.Dropout(data=conv3_2, p=0.25)

        # Block #4: second CONV => relu => POOL layer set
        conv4_1 = mx.sym.Convolution(data=do3, kernel=(3, 3), pad=(1, 1), num_filter=256) # 29 29 256
        act4_1 = mx.sym.Activation(data=conv4_1, act_type="relu")
        bn4_1 = mx.sym.BatchNorm(data=act4_1)
        pool4 = mx.sym.Pooling(data=bn4_1, pool_type="max", kernel=(3,3), stride=(2,2)) # 13 13 256
        do4 = mx.sym.Dropout(data=pool4, p=0.25)

        # Block #5: second CONV => relu => POOL layer set
        conv5_1 = mx.sym.Convolution(data=do4, kernel=(3, 3), pad=(1, 1), num_filter=256) # 13 13 256
        act5_1 = mx.sym.Activation(data=conv5_1, act_type="relu")
        bn5_1 = mx.sym.BatchNorm(data=act5_1)
        conv5_2 = mx.sym.Convolution(data=bn5_1, kernel=(3, 3), stride=(2,2), num_filter=256) # 6 6 256
        conv5_3 = mx.sym.Convolution(data=conv5_2, kernel=(3,3), num_filter=128) # 4 4 128
        do5 = mx.sym.Dropout(data=conv5_3, p=0.25) 

        # Block #6: second set of FC => relu layers
        flatten = mx.sym.Flatten(data=do5)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=512) # 
        act6_1 = mx.sym.Activation(data=fc1, act_type="relu")
        bn6_1 = mx.sym.BatchNorm(data=act6_1)
        do6 = mx.sym.Dropout(data=bn6_1, p=0.5)

        # softmax classifier
        fc2 = mx.sym.FullyConnected(data=do6, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc2, name="softmax")
        # return the network architecture
        return model

def one_off_callback(trainIter, testIter, oneOff, ctx):
    def _callback(iterNum, sym, arg, aux):
        # construct a model for the symbol so we can make predictions
        # on our data
        model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(data_shapes=testIter.provide_data, label_shapes=testIter.provide_label)
        model.set_params(arg, aux)

        # compute one-off metric for both the training and testing
        # data
        trainMAE = _compute_one_off(model, trainIter, oneOff)
        testMAE = _compute_one_off(model, testIter, oneOff)

        # log the values
        logging.info("Epoch[{}] Train-one-off={:.5f}".format(iterNum,trainMAE))
        logging.info("Epoch[{}] Test-one-off={:.5f}".format(iterNum,testMAE))
    # return the callback method
    return _callback

def _compute_one_off(model, dataIter, oneOff):
    # initialize the total number of samples along with the
    # number of correct (maximum of one off) classifications
    total = 0
    correct = 0
    # loop over the predictions of batches
    for (preds, _, batch) in model.iter_predict(dataIter):
        # convert the batch of predictions and labels to NumPy
        # arrays
        predictions = preds[0].asnumpy().argmax(axis=1)
        labels = batch.label[0].asnumpy().astype("int")
        # loop over the predicted labels and ground-truth labels
        # in the batch
        for (pred, label) in zip(predictions, labels):
            # if correct label is in the set of "one off"
            # predictions, then update the correct counter
            if label in oneOff[pred]:
                correct += 1
            # increment the total number of samples
            total += 1

        # finish computing the one-off metric
        return correct / float(total)
def buildOneOffMappings(le):
    # sort the class labels in ascending order (according to age) and initialize the one off mappings for computing accuracy
    print("####",type(le.classes_))
    #classes = sorted(le.classes_, key=lambda x: int(x.decode("utf-8").split("_")[0]))
    classes = sorted(le.classes_, key=lambda x: int(x.split("_")[0]))
    oneOff = {}

    # loop over the index and name of the (sorted) class labels
    for (i, name) in enumerate(classes):
        # determine the index of the *current* class label name
        # in the *label encoder* (unordered) list, then
        # initialize the index of the previous and next age
        # groups adjacent to the current label
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

args = {"checkpoints":"/kaggle/working/checkpoints","prefix":"expt1","start_epoch":0}

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG, filename="training_{}.log".format(args["start_epoch"]), filemode="w")

# determine the batch and load the mean pixel values
#batchSize = config.BATCH_SIZE * config.NUM_DEVICES
batchSize = 256
means = {"R": 116.92359230120708, "G": 89.59216232333138, "B": 79.98195820734759}

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
    path_imgrec = "/home/hrushikesh/images/adience/rec/age_train.rec",
    data_shape = (3, 59, 59),
    batch_size = batchSize,
    rand_crop = True,
    rand_mirror = True,
    rotate = 7,
    mean_r = means["R"],
    mean_g = means["G"],
    mean_b = means["B"],
    preprocess_threads = 2)

# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
    path_imgrec = '/home/hrushikesh/images/adience/rec/age_val.rec',
    data_shape=(3, 59, 59),
    batch_size = batchSize,
    mean_r = means["R"],
    mean_g = means["G"],
    mean_b = means["B"])

# initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=1e-3, momentum=0.9, wd=0.0005, rescale_grad=1.0 / batchSize)

# construct the checkpoints path, initialize the model argument and auxiliary parameters
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
    # build the LeNet architecture
    print("[INFO] building network...")
    model = MxAlexNet.build(8)

# otherwise, a specific checkpoint was supplied
else:
    # load the checkpoint from disk
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    (model, argParams, auxParams) = mx.model.load_checkpoint(
    checkpointsPath, args["start_epoch"])

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model,
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    num_epoch=100,
    begin_epoch=args["start_epoch"])

# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

DATASET_MEAN = "age"
# check to see if the one-off accuracy callback should be used
if DATASET_MEAN == "age":
    # load the label encoder, then build the one-off mappings for
    # computing accuracy
    le = pickle.loads(open('/home/hrushikesh/dl4cv/age_gender/output/age_le.cpickle', "rb").read())
    oneOff = buildOneOffMappings(le)
    epochEndCBs.append(one_off_callback(trainIter, valIter, oneOff, mx.gpu(0)))

# train the network
print("[INFO] training network...")
model.fit(
    X=trainIter,
    eval_data=valIter,
    eval_metric=metrics,
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs)
