from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K #to access the .json file
from tensorflow.keras.regularizers import l2

class My_Model:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):

        model = Sequential()
        inputShape=(height,width,depth)
        chanDim=-1

        if K.image_data_format() == "channels_first":
            inputShape=(depth,height,width)
            chanDim=1

        # Block #1: Conv => elu => POOL layer set
        model.add(Conv2D(32, (5,5), padding="same", input_shape=inputShape, kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (5,5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #2: Conv => elu => POOL layer set
        model.add(Conv2D(64, (3,3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #3: Conv => elu => POOL layer set
        model.add(Conv2D(128, (3,3), strides=(2,2), kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #4: Conv => elu => POOL layer set
        model.add(Conv2D(256, (3,3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        # Block #5: Conv => elu => POOL layer set
        model.add(Conv2D(256, (3,3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3,3), strides=(2,2), kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), strides=(2,2), kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #6: set of FX => elu 
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=l2(reg)))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes,kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        # return the network architecture
        return model

