import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,ReLU
from tensorflow.keras.optimizers import SGD

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
