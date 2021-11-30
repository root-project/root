#ifndef TMVA_SOFIE_ROPERATOR_SLICE
#define TMVA_SOFIE_ROPERATOR_SLICE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

// slice operator

template <typename T, typename IType>
class ROperator_Slice final : public ROperator
{

private:

   std::string fNData;        // input data tensor name
   std::string fNOutput;      // output data name
   std::vector<std::string> fNames;       // tensor names for meta(axis) information
   std::vector<size_t> fShapeInput;     // input shape data
   std::vector<size_t> fShapeOutput;   // output shape data 
   // saved Start/End.Steps are corrected from initial ONNX for negative/default values
   // and are available for each axis
   std::vector<size_t> fStart;         // starting values of slices 
   std::vector<size_t> fEnd;           // End values of slices
   std::vector<size_t> fSteps;         // step values of slices


public:

   ROperator_Slice(){}
   ROperator_Slice(std::string nameData, std::vector<std::string> names, std::string nameOutput)
      : fNData(UTILITY::Clean_name(nameData)), 
      fNOutput(UTILITY::Clean_name(nameOutput))
   {
    fNames.resize(4);
    for (size_t i = 0; i < names.size(); ++i) {
        fNames[i] = UTILITY::Clean_name(names[i]);
    }
    // case names size is 3 check if steps is provided
    // instead of axis. Keep in fNames always the same order 
    if (names.size() == 3 && names[2] != "axes") {
        fNames[3] = fNames[2];
        fNames[2] = "";
    }

   }


   // output type is same as input 
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      auto ret = std::vector<ETensorType>(1, input[0]);
      return ret;
   }

   // output shape
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto & input_shape = input[0]; 
       // assume dimension of output shape is SAME AS INPUT !
      std::vector<std::vector<size_t>> ret(1, input_shape);
      auto & output_shape = ret[0];
      for (size_t i = 0; i < input_shape.size(); i++) { 
          output_shape[i] = (fEnd[i]-fStart[i])/ fSteps[i];
      }
      return ret;
   }


   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA Slice Op Input Tensor is not found in model");
      }

      std::vector<std::vector<size_t>> shapes;
      fShapeInput = model.GetTensorShape(fNData);
      shapes.push_back(fShapeInput);

      std::vector<std::vector<IType>> itensors(4);
      // loop on the extra 2 or 3 or 4 inputs
      for (size_t i = 0; i < fNames.size(); ++i) {
        if (!fNames[i].empty()) { 
         std::cout << " i " << i << " getting data for tensor " << fNames[i] << std::endl;
         auto dptr = model.GetInitializedTensorData(fNames[i]);
         auto tensor = static_cast<IType *>(dptr.get());
         auto vec = model.GetTensorShape(fNames[i]);
         assert(vec.size() == 1);
         itensors[i] = std::vector<IType>(tensor, tensor + vec[0]);
        }
        //  size_t n = vec[0]; // size of shape input tensor
        //  std::vector<size_t> descShape(n);
        //  std::copy(tensor, tensor + n, descShape.begin());
        //  shapes.push_back(descShape);
      }

      size_t dim = fShapeInput.size();
    //   auto axes = std::vector<size_t>(dim);
    //   for (size_t i = 0; i < axes.size(); i++) axes[i] = i;
      
      fSteps = std::vector<size_t>(dim, 1);
      fStart = std::vector<size_t>(dim, 0);
      fEnd = fShapeInput;

     
      //if (fNames.size() == 2) {
      // case of default values 2 or 3 inputs


      auto istart = itensors[0];
      auto iend = itensors[1];
      auto iaxes = itensors[2];
      auto isteps  = itensors[3];
      
      // make tensor axis
      // if size is 0 tensor axis is missing and use defaults
      if (iaxes.size() > 0) {  
        // axis tensor is present
        assert(fNames[2] == "axes");
        for (size_t i = 0; i < iaxes.size(); i++) {
            // negative axes - they count from the back
            if (iaxes[i] < 0) iaxes[i] = dim + iaxes[i];
            size_t jaxis = static_cast<size_t>(iaxes[i]);
            assert(jaxis < dim - 1);
            // find start/end/step for given axis
            int start = (istart[i] > 0) ? istart[i] : dim + istart[i];
            if (start < 0) start = 0;
            if (start > static_cast<int>(fShapeInput[jaxis]))
               start = fShapeInput[jaxis];
            fStart[jaxis] = start;
            int ie = (iend[i] > 0) ? iend[i] : dim + iend[i];
            if (ie < 0)  ie = 0;
            if (ie > static_cast<int>(fShapeInput[jaxis]))
               ie = fShapeInput[jaxis];
            fEnd[jaxis] = ie;
            if (isteps.size() > 0) {
               if (isteps[i] < 0) {
                  // to be done
                  throw std::runtime_error("TMVA Slice Op : negative steps not supported");
               }
               fSteps[jaxis] = isteps[i];
               assert(fSteps[jaxis] > 0 && fSteps[jaxis] < fShapeInput[jaxis]);
            }
        }      
      }
      
      fShapeOutput = ShapeInference({fShapeInput})[0];
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeOutput.empty()){
         throw std::runtime_error("TMVA SOFIE Slice Op called to Generate without being initialized first");
      }

      // output of Slice is same as input
      size_t length = ConvertShapeToLength(fShapeOutput);
      if (length != ConvertShapeToLength(fShapeInput)) {
         throw std::runtime_error("TMVA SOFIE Slice Op : wrong output shape - is " 
         + ConvertShapeToString(fShapeOutput) + " and input is " + ConvertShapeToString(fShapeInput));
      }
      for (auto &i : fShapeOutput) {
         length *= i;
      }
      std::stringstream out;
      //std::string opName = "Slice";

      out << SP << "///------- Slice operator\n" << std::endl;
      // loop on the dimensions depending no the orders 
      size_t ndim = fShapeInput.size();
      std::vector<size_t> strides(ndim,1);
      for (int i = int(ndim-2); i >=0 ; i--) {
          strides[i] = strides[i+1]*fShapeInput[i+1];
      }
    
      std::string MSP = SP;
      for (size_t idim = 0; idim < ndim; idim++) {
        out << MSP << "for (size_t i" << idim << " = " << fStart[idim] <<  "; i" << idim << " < " << fEnd[idim] 
            << "; i" << idim << "+= " << fSteps[idim] << ") {\n"; 
        MSP += SP;
        if (idim < ndim-1) out << MSP << "size_t stride" << idim << " = " << strides[idim] << "*i" << idim << ";\n";
      }
      out << MSP << "size_t iInput = ";
      for (size_t idim = 0; idim < ndim-1; idim++) out << " stride" << idim << " + ";
      // here should be step size ?
      out << "i" << ndim-1 << ";\n";
      out << MSP << "fTensor_" << fNOutput << "[iOut++] = fTensor_" <<fNData << "[iInput];\n";
      for (size_t idim = 0; idim < ndim; idim++) {
          MSP = MSP.replace(0,SP.length(),"");
          out << MSP << "}\n";
      }
      
      return out.str();
   }


};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_SLICE
