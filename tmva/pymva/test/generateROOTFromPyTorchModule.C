#include "TMVA/RModelParser_PyTorch.h"

using namespace TMVA::Experimental;

int generateROOTFromPyTorchModule(){
    Py_Initialize();
    std::cout<<"Generating PyTorch nn.Module model...";
    FILE* fp;
    fp = fopen("generatePyTorchModelModule.py", "r");
    PyRun_SimpleFile(fp, "generatePyTorchModelModule.py");
    std::cout<<"OK\n";
    TFile fileWrite("PyTorchModuleModel.root","RECREATE");
    std::vector<size_t> s1{2,6};
    std::vector<std::vector<size_t>> inputShape{s1};
    std::cout<<"Parsing saved PyTorch nn.Module model...";
    SOFIE::RModel model = SOFIE::PyTorch::Parse("PyTorchModelModule.pt",inputShape);
    model.Generate();
    std::cout<<"OK\n";
    std::cout<<"Writing PyTorch nn.Module RModel into a ROOT file...";
    model.Write("model");
    fileWrite.Close();
    std::cout<<"OK\n";
    std::cout<<"Reading the saved ROOT file and generating the header file for RModel...";
    TFile fileRead("PyTorchModuleModel.root","READ");
    SOFIE::RModel *modelPtr;
    fileRead.GetObject("model",modelPtr);
    fileRead.Close();
    modelPtr->OutputGenerated("PyTorchModuleModel.hxx");
    std::cout<<"OK\n";
    return 0;
}
