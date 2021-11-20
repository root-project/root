#ifndef TMVA_SOFIE_ROPERATOR_RESHAPE
#define TMVA_SOFIE_ROPERATOR_RESHAPE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{




template <typename T>
class ROperator_Reshape final : public ROperator
{

private:
   std::vector<int_t> fAttrPerm;  

   std::string fNData;        // input data tensor name
   std::string fNShape;       // reshape tensor name
   std::string fNOutput;               // output tensor name
   std::vector<size_t> fShapeData;     // input shape data
   std::vector<size_t> fShapeOutput;   // output shape data

   bool fFlatten = false;

public:

   ROperator_Reshape(){}
   ROperator_Reshape(std::vector<int_t> attr_perm, std::string nameData, std::string nameShape, std::string nameOutput):
      fAttrPerm(attr_perm), fNData(UTILITY::Clean_name(nameData)), fNShape(UTILITY::Clean_name(nameShape)), fNOutput(UTILITY::Clean_name(nameOutput)) {
   }

   ROperator_Reshape(std::string nameData, std::string nameShape, std::string nameOutput)
      : fNData(UTILITY::Clean_name(nameData)) , fNOutput(UTILITY::Clean_name(nameOutput))
   {
      if (nameShape.empty())
         fFlatten = true;
      else
         fNShape = UTILITY::Clean_name(nameShape);
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      auto ret = std::vector<ETensorType>(1, input[0]);
      return ret;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      std::vector<std::vector<size_t>> ret;
      if (!fFlatten) { 
         if (input.size() != 2) throw std::runtime_error("TMVA SOFIE Reshape Op needs 2 input tensors");
         ret.push_back(input[1]);
      }
      else { 
         //flattenig case
         size_t inputSize = ConvertShapeToLength(input[0]);
         size_t b = input[0][0];
         std::vector<size_t> newShape = {b, inputSize / b};
         ret.push_back(newShape);
      }
      return ret;
   }


   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA Reshape Op Input Tensor is not found in model");
      }
      fShapeData = model.GetTensorShape(fNData);
      // fShapeOutput = model.GetTensorShape(fNShape);
      if (!fFlatten)
        fShapeOutput = ShapeInference({fShapeData,model.GetTensorShape(fNShape)})[0];
      else
         fShapeOutput = ShapeInference({fShapeData})[0];

      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
   
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeData.empty() || fShapeOutput.empty()){
         throw std::runtime_error("TMVA SOFIE Reshape Op called to Generate without being initialized first");
      }

      // output of reshape is same as input
      int length = 1;
      for (auto &i : fShapeOutput) {
         length *= i;
      }
      std::stringstream out;
      out << "\t"
          << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t"
          << "tensor_" << fNOutput << "[id] = tensor_" << fNData << "[id];\n";
      out << "\t}\n";
      return out.str();
   }


};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RESHAPE
