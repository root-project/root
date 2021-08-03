#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimental;

int generateROOTFromKerasSequential(){
    Py_Initialize();
    std::cout<<"Generating Keras Sequential API model...";
    FILE* fp;
    fp = fopen("generateKerasModelSequential.py", "r");
    PyRun_SimpleFile(fp,"generateKerasModelSequential.py");
    fclose(fp);
    std::cout<<"OK\n";
    TFile fileWrite("KerasSequentialModel.root","RECREATE");
    std::cout<<"Parsing saved Keras Sequential API model...";
    SOFIE::RModel model = SOFIE::PyKeras::Parse("KerasModelSequential.h5");
    model.Generate();
    std::cout<<"OK\n";
    std::cout<<"Writing Keras Sequential API RModel into a ROOT file...";
    model.Write("model");
    fileWrite.Close();
    std::cout<<"OK\n";
    TFile fileRead("KerasSequentialModel.root","READ");
    SOFIE::RModel *modelPtr;
    fileRead.GetObject("model",modelPtr);
    fileRead.Close();
    std::cout<<"Generating header file from SOFIE RModel...";
    modelPtr->OutputGenerated("KerasSequentialModel.hxx");
    std::cout<<"OK\n";
    return 0;
}
