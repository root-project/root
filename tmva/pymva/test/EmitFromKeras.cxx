// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the RModelParser_Keras.
//           The program is run when the target 'TestRModelParserKeras' is built.
//           The program generates the required .hxx file after parsing a Keras .h5 file into a RModel object.


#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_Keras.h"

#include <Python.h>

using namespace TMVA::Experimental::SOFIE;

int main(){

   Py_Initialize();

   //Emitting header file for Keras Sequential API Model
   FILE* fKerasModels;
   fKerasModels = fopen("generateKerasModels.py", "r");
   PyRun_SimpleFile(fKerasModels, "generateKerasModels.py");

   //Emitting header file for Keras Sequential Model
   RModel modelSequential = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelSequential.h5");
   modelSequential.Generate();
   modelSequential.OutputGenerated("KerasSequentialModel.hxx");

   //Emitting header file for Keras Functional API Model
   RModel modelFunctional = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelFunctional.h5");
   modelFunctional.Generate();
   modelFunctional.OutputGenerated("KerasFunctionalModel.hxx");

   //Emitting header file for Keras BatchNorm Model
   RModel modelBatchNorm = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelBatchNorm.h5");
   modelBatchNorm.Generate();
   modelBatchNorm.OutputGenerated("KerasBatchNormModel.hxx");

   return 0;
}
