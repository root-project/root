import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Activation,ReLU,BatchNormalization,Conv2D,Reshape
from tensorflow.keras.optimizers import SGD

def generateFunctionalModel():
    input=Input(shape=(8,),batch_size=2)
    x=Dense(16)(input)
    x=Activation('relu')(x)
    x=Dense(32)(x)
    x=ReLU()(x)
    output=Dense(4, activation='selu')(x)
    model=Model(inputs=input,outputs=output)

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(2,8)
    y_train=randomGenerator.rand(2,4)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=2)
    model.save('KerasModelFunctional.h5')

def generateSequentialModel():
    model=Sequential()
    model.add(Dense(8,batch_size=4))
    model.add(ReLU())
    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(4,8)
    y_train=randomGenerator.rand(4,6)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=4)
    model.save('KerasModelSequential.h5')

def generateBatchNormModel():
    model=Sequential()
    model.add(Dense(4,batch_size=2))
    model.add(BatchNormalization())
    model.add(Dense(2))

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(4,4)
    y_train=randomGenerator.rand(4,2)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=2)
    model.save('KerasModelBatchNorm.h5')

def generateConv2DModel_ValidPadding():
    model=Sequential()
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=(4,4,1), padding="valid"))

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(1,4,4,1)
    y_train=randomGenerator.rand(1,2,2,8)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=2)
    model.save('KerasModelConv2D_Valid.h5')

def generateConv2DModel_SamePadding():
    model=Sequential()
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=(4,4,1), padding="same"))

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(1,4,4,1)
    y_train=randomGenerator.rand(1,4,4,8)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=2)
    model.save('KerasModelConv2D_Same.h5')
    
def generateReshapeModel():
    model = Sequential()
    model.add(Dense(3, input_shape=(4,)))
    model.add(Reshape((3, 1)))

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(1,4)
    y_train=randomGenerator.rand(1,3,1)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=2)
    model.save('KerasModelReshape.h5')


generateFunctionalModel()
generateSequentialModel()
generateBatchNormModel()
generateConv2DModel_ValidPadding()
generateConv2DModel_SamePadding()
generateReshapeModel()
