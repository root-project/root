// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the RModelParser_PyTorch.
//           The program is run when the target 'TestRModelParserPyTorch' is built.
//           The program generates the required .hxx file after parsing a PyTorch .pt file into a RModel object.


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_PyTorch.h"

using namespace TMVA::Experimental::SOFIE;

int main(){

   Py_Initialize();
   //Emitting header file for PyTorch nn.Module
   FILE* fPyTorchModule;
   fPyTorchModule = fopen("generatePyTorchModelModule.py", "r");
   PyRun_SimpleFile(fPyTorchModule, "generatePyTorchModelModule.py");
   std::vector<size_t> inputTensorShapeModule{2,6};
   std::vector<std::vector<size_t>> inputShapesModule{inputTensorShapeModule};
   RModel modelModule = TMVA::Experimental::SOFIE::PyTorch::Parse("PyTorchModelModule.pt",inputShapesModule);
   modelModule.Generate();
   modelModule.OutputGenerated("PyTorchModuleModel.hxx");

   //Emitting header file for PyTorch nn.Sequential
   FILE* fPyTorchSequential;
   fPyTorchSequential = fopen("generatePyTorchModelSequential.py", "r");
   PyRun_SimpleFile(fPyTorchSequential, "generatePyTorchModelSequential.py");
   std::vector<size_t> inputTensorShapeSequential{2,4};
   std::vector<std::vector<size_t>> inputShapesSequential{inputTensorShapeSequential};
   RModel modelSequential = TMVA::Experimental::SOFIE::PyTorch::Parse("PyTorchModelSequential.pt",inputShapesSequential);
   modelSequential.Generate();
   modelSequential.OutputGenerated("PyTorchSequentialModel.hxx");

   return 0;
}
