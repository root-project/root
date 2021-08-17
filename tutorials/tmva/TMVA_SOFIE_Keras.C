/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example for the parsing of Keras .h5 file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta

using namespace TMVA::Experimental;

TString pythonSrc = "\
import os\n\
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n\
\n\
import keras\n\
import numpy as np\n\
from keras.models import Model\n\
from keras.layers import Input,Dense,Activation,ReLU\n\
from keras.optimizers import SGD\n\
\n\
input=Input(shape=(8,),batch_size=2)\n\
x=Dense(16)(input)\n\
x=Activation('relu')(x)\n\
x=Dense(32,activation='relu')(x)\n\
x=Dense(4)(x)\n\
output=ReLU()(x)\n\
model=Model(inputs=input,outputs=output)\n\
\n\
randomGenerator=np.random.RandomState(0)\n\
x_train=randomGenerator.rand(2,8)\n\
y_train=randomGenerator.rand(2,4)\n\
\n\
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))\n\
model.fit(x_train, y_train, epochs=10, batch_size=2)\n\
model.save('KerasModelFunctional.h5')\n";


void TMVA_SOFIE_Keras(){

    //Running the Python script to generate Keras .h5 file
    Py_Initialize();
    PyRun_SimpleString(pythonSrc);

    //Parsing the saved Keras .h5 file into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse("KerasModelFunctional.h5");

    //Generating inference code
    model.Generate();
    model.OutputGenerated("KerasModelFunctional.hxx");

    //Printing required input tensors
    model.PrintRequiredInputTensors();

    //Printing initialized tensors (weights)
    std::cout<<"\n\n";
    model.PrintInitializedTensors();

    //Printing intermediate tensors
    std::cout<<"\n\n";
    model.PrintIntermediateTensors();

    //Checking if tensor already exist in model
    std::cout<<"\n\nTensor \"dense2bias0\" already exist: "<<std::boolalpha<<model.CheckIfTensorAlreadyExist("dense2bias0")<<"\n\n";
    std::vector<size_t> tensorShape = model.GetTensorShape("dense2bias0");
    std::cout<<"Shape of tensor \"dense2bias0\": ";
    for(auto& it:tensorShape){
        std::cout<<it<<",";
    }
    std::cout<<"\n\nData type of tensor \"dense2bias0\": ";
    SOFIE::ETensorType tensorType = model.GetTensorType("dense2bias0");
    std::cout<<SOFIE::ConvertTypeToString(tensorType);

    //Printing generated inference code
    std::cout<<"\n\n";
    model.PrintGenerated();
}
