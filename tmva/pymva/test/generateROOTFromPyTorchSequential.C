#include "TMVA/RModelParser_PyTorch.h"

using namespace TMVA::Experimental;

int generateROOTFromPyTorchSequential(){
    Py_Initialize();
    std::cout<<"Generating PyTorch nn.Sequential model...";
    FILE* fp;
    fp = fopen("generatePyTorchModelSequential.py", "r");
    PyRun_SimpleFile(fp, "generatePyTorchModelSequential.py");
    std::cout<<"OK\n";
    TFile fileWrite("PyTorchSequentialModel.root","RECREATE");
    std::vector<size_t> s1{2,4};
    std::vector<std::vector<size_t>> inputShape{s1};
    std::cout<<"Parsing saved PyTorch nn.Sequential model...";
    SOFIE::RModel model = SOFIE::PyTorch::Parse("PyTorchModelSequential.pt",inputShape);
    model.Generate();
    std::cout<<"OK\n";
    std::cout<<"Writing PyTorch nn.Sequential RModel into a ROOT file...";
    model.Write("model");
    fileWrite.Close();
    std::cout<<"OK\n";
    std::cout<<"Reading the saved ROOT file and generating the header file for RModel...";
    TFile fileRead("PyTorchSequentialModel.root","READ");
    SOFIE::RModel *modelPtr;
    fileRead.GetObject("model",modelPtr);
    fileRead.Close();
    modelPtr->OutputGenerated("PyTorchSequentialModel.hxx");
    std::cout<<"OK\n";
    return 0;
}
