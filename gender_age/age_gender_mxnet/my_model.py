import mxnet as mx

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

