// Author: Sanjiban Sengupta, 2022
// Description:
//           This is to test the usability of the Custom Operator
//           in TMVA SOFIE. The program is run when the
//           target 'TestCustomOp' is built. The program
//           first creates a Keras .h5 file, parses it and adds
//           a custom operator and then generates the inference header 
//           file. It also creates another Keras .h5 file with the library 
//           definition of the custom operator included to test the equality.

#include "TMVA/RModel.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModelParser_Keras.h"

using namespace TMVA::Experimental::SOFIE;

int main(){
    
    //Parsing the saved Keras .h5 file into RModel object
    RModel model = PyKeras::Parse("KerasModelForCustomOp.h5");
    model.Generate();

    std::unique_ptr<ROperator> op;
    op.reset(new ROperator_Custom<float>(/*OpName*/ "Double", /*input tensor names*/{"dense9Relu0"}, 
                                        /*output tensor names*/{"output"}, /*output shapes*/{{1,4}},
                                        /*header file name with the compute function*/ "double_op.hxx"));

    // adding the custom op in the model
    model.AddOperator(std::move(op));
    
    // Generating inference code again after adding the custom operator
    model.Generate();
    model.OutputGenerated("KerasModelWithCustomOp.hxx");

    return 0;
}