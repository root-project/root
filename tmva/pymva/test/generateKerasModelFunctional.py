import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Activation,ReLU
from keras.optimizers import SGD

input=Input(shape=(8,),batch_size=2)
x=Dense(16)(input)
x=Activation('relu')(x)
x=Dense(32,activation='relu')(x)
x=Dense(4)(x)
output=ReLU()(x)
model=Model(inputs=input,outputs=output)

x_train=np.random.rand(2,8)
y_train=np.random.rand(2,4)

model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
model.fit(x_train, y_train, epochs=10, batch_size=2)
model.save('KerasModelFunctional.h5')
