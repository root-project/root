/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro provides a simple example for the parsing of ONNX files into
/// RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta
using namespace TMVA::Experimental;
void TMVA_SOFIE_ONNX(std::string inputFile = ""){
   if (inputFile.empty() )
      inputFile = std::string(gROOT->GetTutorialsDir()) + "/machine_learning/Linear_16.onnx";
    //Creating parser object to parse ONNX files
    SOFIE::RModelParser_ONNX parser;
    SOFIE::RModel model = parser.Parse(inputFile, true);
    //Generating inference code
    model.Generate();
    // write the code in a file (by default Linear_16.hxx and Linear_16.dat)
    model.OutputGenerated();
    //Printing required input tensors
    model.PrintRequiredInputTensors();
    //Printing initialized tensors (weights)
    std::cout<<"\n\n";
    model.PrintInitializedTensors();
    //Printing intermediate tensors
    std::cout<<"\n\n";
    model.PrintIntermediateTensors();
    //Checking if tensor already exist in model
    const std::string tensorName = "0weight";
    bool tensorExists = model.CheckIfTensorAlreadyExist(tensorName);
    std::cout<<"\n\nTensor \""<<tensorName<<"\" already exist: "<<std::boolalpha<<tensorExists<<"\n\n";
    if (tensorExists) {
        std::vector<size_t> tensorShape = model.GetTensorShape(tensorName);
        std::cout<<"Shape of tensor \""<<tensorName<<"\": ";
        for(auto& it:tensorShape){
            std::cout<<it<<",";
        }
        std::cout<<"\n\nData type of tensor \""<<tensorName<<"\": ";
        SOFIE::ETensorType tensorType = model.GetTensorType(tensorName);
        std::cout<<SOFIE::ConvertTypeToString(tensorType);
    } else {
        std::cout<<"Tensor \""<<tensorName<<"\" not found in model. Skipping shape/type queries.\n";
    }
    //Printing generated inference code
    std::cout<<"\n\n";
    model.PrintGenerated();
}
