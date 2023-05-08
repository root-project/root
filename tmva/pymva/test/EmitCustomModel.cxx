// Author: Sanjiban Sengupta, 2022
// Description:
//           This is to test the Custom Operator
//           in TMVA SOFIE. The program is run when the
//           target 'TestCustomOp' is built. The program
//           first creates a Keras .h5 file, parses it and adds
//           a custom operator and then generates the inference header 
//           file. During the test, the custom operator is added as a 
//           Lambda layer in the Keras model and is tested for equality.

#include "Python.h"

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimental::SOFIE;

const char* pythonSrc = "\
import os\n\
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n\
\n\
import numpy as np\n\
from tensorflow.keras.models import Sequential\n\
from tensorflow.keras.layers import Dense\n\
from tensorflow.keras.optimizers import SGD\n\
\n\
model = Sequential()\n\
model.add(Dense(4, activation='relu'))\n\
\n\
randomGenerator=np.random.RandomState(0)\n\
x_train=randomGenerator.rand(1,8)\n\
y_train=randomGenerator.rand(1,4)\n\
\n\
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))\n\
model.fit(x_train, y_train, epochs=10, batch_size=1)\n\
model.save('KerasModelForCustomOp.h5')\n";

int main(){
   Py_Initialize();

   //Emitting header file for Keras Model
   PyRun_SimpleString(pythonSrc);

    //Parsing the saved Keras .h5 file into RModel object
    RModel model = PyKeras::Parse("KerasModelForCustomOp.h5");
    model.Generate();

    std::unique_ptr<ROperator> op;
    op.reset(new ROperator_Custom<float>(/*OpName*/ "Scale_by_2", /*input tensor names*/model.GetOutputTensorNames(), 
                                        /*output tensor names*/{"output"}, /*output shapes*/{{1,4}},
                                        /*header file name with the compute function*/ "scale_by_2_op.hxx"));

    // adding the custom op in the model
    model.AddOperator(std::move(op));
    
    // Generating inference code again after adding the custom operator
    model.Generate();
    model.OutputGenerated("KerasModelWithCustomOp.hxx");

    return 0;
}