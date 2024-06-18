#ifndef TMVA_SOFIE_ROPERATOR_Tile
#define TMVA_SOFIE_ROPERATOR_Tile

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator
{

private:

   std::string fNInput;
   std::string fNRepeats;
   std::string fNY;
   std::vector<size_t> fShapeY;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      std::vector<size_t> ret = {input[0]};
      for(size_t i=0; i < input[1].size(); i++) {
            ret[i]=ret[i]*input[1][i];
      }
      return {ret};
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      std::vector<size_t>input_shape=model.GetTensorShape(fNInput);
      std::vector<size_t>repeat_shape=model.GetTensorShape(fNRepeats);
      fShapeY = ShapeInference({input_shape,repeat_shape})[0];
      for(auto i : fShapeY)std::cout<<i<<" ";
      std::cout<<std::endl;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Tile operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeY);
      out << "\n//------ TILE\n";
    //   out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
    //   out << SP << SP << "tensor_" << fNY << "[id] = std::tanh(tensor_" << fNX << "[id]);\n";
    //   out << SP << "}\n";
      return out.str();
   }

//    std::vector<std::string> GetStdLibs() { return { std::string("cmath") };}
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Tile
