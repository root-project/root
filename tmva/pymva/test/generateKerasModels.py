import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Activation,ReLU,BatchNormalization
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
    model.add(Dense(8,batch_size=4))
    model.add(BatchNormalization())

    randomGenerator=np.random.RandomState(0)
    x_train=randomGenerator.rand(4,8)
    y_train=randomGenerator.rand(4,8)

    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(x_train, y_train, epochs=10, batch_size=4)
    model.save('KerasModelBatchNorm.h5')

generateFunctionalModel()
generateSequentialModel()
generateBatchNormModel()