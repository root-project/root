#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimental;

int generateROOTFromKerasFunctional(){
    Py_Initialize();
    std::cout<<"Generating Keras Functional API model...";
    FILE* fp;
    fp = fopen("generateKerasModelFunctional.py", "r");
    PyRun_SimpleFile(fp,"generateKerasModelFunctional.py");
    fclose(fp);
    std::cout<<"OK\n";
    TFile fileWrite("KerasFunctionalModel.root","RECREATE");
    std::cout<<"Parsing saved Keras Functional API model...";
    SOFIE::RModel model = SOFIE::PyKeras::Parse("KerasModelFunctional.h5");
    model.Generate();
    std::cout<<"OK\n";
    std::cout<<"Writing Keras Functional API RModel into a ROOT file...";
    model.Write("model");
    fileWrite.Close();
    std::cout<<"OK\n";
    TFile fileRead("KerasFunctionalModel.root","READ");
    SOFIE::RModel *modelPtr;
    fileRead.GetObject("model",modelPtr);
    fileRead.Close();
    std::cout<<"Generating header file from SOFIE RModel...";
    modelPtr->OutputGenerated("KerasFunctionalModel.hxx");
    std::cout<<"OK\n";
    return 0;
}
