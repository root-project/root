// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the RModelParser_PyTorch.
//           The program is run when the target 'TestRModelParserPyTorch' is built.
//           The program generates the required .hxx file after parsing a PyTorch .pt file into a RModel object.


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimental::SOFIE;

int main(){

   Py_Initialize();

   //Emitting header file for Keras Sequential API Model
   FILE* fKerasSequential;
   fKerasSequential = fopen("generateKerasModelSequential.py", "r");
   PyRun_SimpleFile(fKerasSequential, "generateKerasModelSequential.py");
   RModel modelSequential = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelSequential.h5");
   modelSequential.Generate();
   modelSequential.OutputGenerated("KerasSequentialModel.hxx");

   //Emitting header file for Keras Functional API Model
   FILE* fKerasFunctional;
   fKerasFunctional = fopen("generateKerasModelFunctional.py", "r");
   PyRun_SimpleFile(fKerasFunctional, "generateKerasModelFunctional.py");
   RModel modelFunctional = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelFunctional.h5");
   modelFunctional.Generate();
   modelFunctional.OutputGenerated("KerasFunctionalModel.hxx");

   return 0;
}
