// Author: Sanjiban Sengupta, 2021
// Description:
//           This is to test the RModelParser_Keras.
//           The program is run when the target 'TestRModelParserKeras' is built.
//           The program generates the required .hxx file after parsing a Keras .keras file into a RModel object.


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
   RModel modelSequential = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelSequential.keras");
   modelSequential.Generate();
   modelSequential.OutputGenerated("KerasSequentialModel.hxx");

   //Emitting header file for Keras Functional API Model
   RModel modelFunctional = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelFunctional.keras");
   modelFunctional.Generate();
   modelFunctional.OutputGenerated("KerasFunctionalModel.hxx");

   //Emitting header file for Keras BatchNorm Model
   RModel modelBatchNorm = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelBatchNorm.keras");
   modelBatchNorm.Generate();
   modelBatchNorm.OutputGenerated("KerasBatchNormModel.hxx");

#if PY_MAJOR_VERSION >= 3  // parsing of convolutional models supported only for Python3
   // Emitting header file for Keras Conv2D Model with valid padding
   RModel modelConv2D_Valid = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelConv2D_Valid.keras");
   modelConv2D_Valid.Generate();
   modelConv2D_Valid.OutputGenerated("KerasConv2D_Valid.hxx");

   // Emitting header file for Keras Conv2D Model with valid padding
   RModel modelConv2D_Same = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelConv2D_Same.keras");
   modelConv2D_Same.Generate();
   modelConv2D_Same.OutputGenerated("KerasConv2D_Same.hxx");
#endif

   //Emitting header file for Keras model with Reshape layer
   RModel modelReshape = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelReshape.keras");
   modelReshape.Generate();
   modelReshape.OutputGenerated("KerasReshapeModel.hxx");

   //Emitting header file for Keras model with Concatenate layer
   RModel modelConcatenate = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelConcatenate.keras");
   modelConcatenate.Generate();
   modelConcatenate.OutputGenerated("KerasConcatenateModel.hxx");

   // Emitting header file for Keras Binary Op Model
   RModel modelBinaryOp = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelBinaryOp.keras");
   modelBinaryOp.Generate();
   modelBinaryOp.OutputGenerated("KerasBinaryOpModel.hxx");

   //Emitting header file for Keras activation functions model
   RModel modelActivations = TMVA::Experimental::SOFIE::PyKeras::Parse("KerasModelActivations.keras");
   modelActivations.Generate();
   modelActivations.OutputGenerated("KerasActivationsModel.hxx");

   // Emitting header file for Swish activation functions model
   RModel modelSwish = TMVA::Experimental::SOFIE::PyKeras::Parse("swish_model.keras");
   modelSwish.Generate();
   modelSwish.OutputGenerated("KerasSwish_model.hxx");

   return 0;
}
