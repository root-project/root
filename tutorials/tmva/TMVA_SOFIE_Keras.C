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
import numpy as np\n\
from tensorflow.keras.models import Model\n\
from tensorflow.keras.layers import Input,Dense,Activation,ReLU\n\
from tensorflow.keras.optimizers import SGD\n\
\n\
input=Input(shape=(64,),batch_size=4)\n\
x=Dense(32)(input)\n\
x=Activation('relu')(x)\n\
x=Dense(16,activation='relu')(x)\n\
x=Dense(8,activation='relu')(x)\n\
x=Dense(4)(x)\n\
output=ReLU()(x)\n\
model=Model(inputs=input,outputs=output)\n\
\n\
randomGenerator=np.random.RandomState(0)\n\
x_train=randomGenerator.rand(4,64)\n\
y_train=randomGenerator.rand(4,4)\n\
\n\
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))\n\
model.fit(x_train, y_train, epochs=5, batch_size=4)\n\
model.save('KerasModel.h5')\n";


void TMVA_SOFIE_Keras(const char * modelFile = nullptr, bool printModelInfo = true){

    //Running the Python script to generate Keras .h5 file
    TMVA::PyMethodBase::PyInitialize();

    if (modelFile == nullptr) {
        TMacro m;
        m.AddLine(pythonSrc);
        m.SaveSource("make_keras_model.py");
        gSystem->Exec(TMVA::Python_Executable() + " make_keras_model.py");
        modelFile = "KerasModel.h5";
    }

    //Parsing the saved Keras .h5 file into RModel object
    SOFIE::RModel model = SOFIE::PyKeras::Parse(modelFile);


    //Generating inference code
    model.Generate();
    // generate output header. By default it will be modelName.hxx
    model.OutputGenerated();

     if (!printModelInfo) return;

    //Printing required input tensors
    std::cout<<"\n\n";
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
