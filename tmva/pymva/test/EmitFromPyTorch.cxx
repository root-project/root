// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the RModelParser_PyTorch.
//           The program is run when the target 'TestRModelParserPyTorch' is built.
//           The program generates the required .hxx file after parsing a PyTorch .pt file into a RModel object.


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_PyTorch.h"

#include <Python.h>

using namespace TMVA::Experimental::SOFIE;

int main(){

   Py_Initialize();

   // Generating PyTorch models for testing
   FILE* fPyTorchModels;
   fPyTorchModels = fopen("generatePyTorchModels.py", "r");
   PyRun_SimpleFile(fPyTorchModels, "generatePyTorchModels.py");

   //Emitting header file for PyTorch nn.Module
   std::vector<size_t> inputTensorShapeModule{2,6};
   std::vector<std::vector<size_t>> inputShapesModule{inputTensorShapeModule};
   RModel modelModule = TMVA::Experimental::SOFIE::PyTorch::Parse("PyTorchModelModule.pt",inputShapesModule);
   modelModule.Generate();
   modelModule.OutputGenerated("PyTorchModuleModel.hxx");

   //Emitting header file for PyTorch nn.Sequential
   std::vector<size_t> inputTensorShapeSequential{2,4};
   std::vector<std::vector<size_t>> inputShapesSequential{inputTensorShapeSequential};
   RModel modelSequential = TMVA::Experimental::SOFIE::PyTorch::Parse("PyTorchModelSequential.pt",inputShapesSequential);
   modelSequential.Generate();
   modelSequential.OutputGenerated("PyTorchSequentialModel.hxx");

   //Emitting header file for PyTorch Convolution Model
   std::vector<size_t> inputTensorShapeConvolution{5, 6, 5, 5};
   std::vector<std::vector<size_t>> inputShapesConvolution{inputTensorShapeConvolution};
   RModel modelConvolution = TMVA::Experimental::SOFIE::PyTorch::Parse("PyTorchModelConvolution.pt",inputShapesConvolution);
   modelConvolution.Generate();
   modelConvolution.OutputGenerated("PyTorchConvolutionModel.hxx");

   return 0;
}
