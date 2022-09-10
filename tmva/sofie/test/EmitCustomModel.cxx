// Author: Sanjiban Sengupta, 2022
// Description:
//           This is to test the usability of the Custom Operator
//           in TMVA SOFIE. The program is run when the
//           target 'TestCustomOp' is built. The program
//           first creates a Keras .h5 file, parses it and adds
//           a custom operator and then generates the inference header 
//           file. It also creates another Keras .h5 file with the library 
//           definition of the custom operator included to test the equality.

#include "TROOT.h"
#include "TSystem.h"
#include "TMacro.h"
#include "TInterpreter.h"

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimenal::SOFIE;

TString KerasModelBasic = "\
import os\n\
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n\
\n\
import numpy as np\n\
from tensorflow.keras.models import Model\n\
from tensorflow.keras.layers import Input,Dense,Activation,ReLU\n\
from tensorflow.keras.optimizers import SGD\n\
\n\
input=Input(shape=(8,),batch_size=1)\n\
x=Dense(4)(input)\n\
x=Activation('relu')(x)\n\
model=Model(inputs=input,outputs=output)\n\
\n\
randomGenerator=np.random.RandomState(0)\n\
x_train=randomGenerator.rand(1,8)\n\
y_train=randomGenerator.rand(1,4)\n\
\n\
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))\n\
model.fit(x_train, y_train, epochs=5, batch_size=1)\n\
model.save('KerasModelForCustomOp.h5')\n";


int main(){
    
    //Running the Python script to generate Keras .h5 file
    TMVA::PyMethodBase::PyInitialize();

    TMacro m;
    m.AddLine(KerasModelBasic);
    m.SaveSource("make_keras_model_basic.py");
    gSystem->Exec(TMVA::Python_Executable() + " make_keras_model_basic.py");

    //Parsing the saved Keras .h5 file into RModel object
    RModel model = SOFIE::PyKeras::Parse("KerasModelForCustomOp.h5");
    model.Generate();

    std::unique_ptr<ROperator> op;
    op.reset(new ROperator_Custom<float>(/*OpName*/ "Double", /*input tensor names*/{"input"}, 
                                        /*output tensor names*/{"output"}, /*output shapes*/{{1,4}},
                                        /*header file name with the compute function*/ "double_op.hxx"));
    
    // Generating inference code again after adding the custom operator
    model.Generate();
    model.OutputGenerated("KerasModelWithCustomOp.hxx");

    return 0;
}
