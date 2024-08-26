#ifndef TMVA_SOFIE_ROPERATOR_Swish
#define TMVA_SOFIE_ROPERATOR_Swish

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Split final : public ROperator
{

private:

   std::string fNX;
   std::string fNS;
   std::vector<std::string> fNYs;
   std::vector<std::vector<size_t>> fOutputShapes;


public:
   ROperator_Split(){}
   ROperator_Split(const std::string & nameX, const std::string & nameS,  const std::vector<std::string> &  namesY):
      fNX(UTILITY::Clean_name(nameX)), fNS(UTILITY::Clean_name(nameS)){
         fNYs.reserve(namesY.size());
         for (auto & name : namesY)
            fNYs.push_back(UTILITY::Clean_name(name));
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Split Op Input Tensor is not found in model");
      }
      auto inputShape = model.GetTensorShape(fNX);

      // support now splitting only of 1D tensors and assuming tensor can be  split in equal parts
      //int splitAxis = 0;   // assume split with zero axis
      int nsplit = fNYs.size();
      // support now only 1D tensor
      if (inputShape.size() > 1)
         throw std::runtime_error("TMVA SOFIE Split Op supports now only 1D tensors");
      // support only equal splits
      if (inputShape[0] % nsplit != 0)
         throw std::runtime_error("TMVA SOFIE Split Op does not support splitting of " + ConvertShapeToString(inputShape)
            + " into " + std::to_string(nsplit));

      for (size_t i = 0; i < fNYs.size(); i++) {
         std::vector<size_t> outputShape = { inputShape[0]/nsplit };
         model.AddIntermediateTensor(fNYs[i], model.GetTensorType(fNX), outputShape);
         fOutputShapes.push_back(outputShape);  // need for generating code
      }
      if (model.Verbose()) {
         std::cout << "Split - input shape " << ConvertShapeToString(inputShape) << " --> ";
         for (auto & s : fOutputShapes)
            std::cout << ConvertShapeToString(s) << "  ";
         std::cout << std::endl;
      }
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fOutputShapes.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Split called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ Split\n";
      out << "size_t offset = 0;\n";
      for (size_t i = 0; i < fNYs.size(); i++)  {
         int length = ConvertShapeToLength(fOutputShapes[i]);
         out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
            out << SP << SP  << "tensor_" << fNYs[i] << "[id] = tensor_" << fNX <<"[offset+id];\n";
         out << SP << "}\n";
         if (i < fNYs.size()-1) out << SP << "offset += " << length << ";\n";
      }
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Swish
