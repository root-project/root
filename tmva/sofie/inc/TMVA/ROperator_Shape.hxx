#ifndef TMVA_SOFIE_ROPERATOR_Shape
#define TMVA_SOFIE_ROPERATOR_Shape

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include<sstream>
#include<vector>
#include <iterator>
#include<string>
namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator_Shape final : public ROperator
{

private:

   /* Attributes*/
   int fStart = 0;  // default is beginning
   int fEnd = 0; // default is input length (all input tensor shape included)
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::vector<size_t> fOutput_shape;

public:
   ROperator_Shape(){}
   ROperator_Shape(int start, int end, std::string nameX, std::string nameY):
   fStart(start) ,fEnd(end), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      std::vector<std::vector<size_t>>  ret;
      ret[0].push_back(input[0].size());
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Shape Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      size_t length = ConvertShapeToLength(fShape);
      fStart = std::max(fStart,(int) -length);
      fStart = std::min(fStart,(int) length);
      if (fStart < 0) fStart += length;
      fEnd = std::max(fEnd,(int) -length);
      fEnd = std::min(fEnd, (int) length);
      if (fEnd < 0) fEnd += length;
      if (fEnd > fStart)
         fOutput_shape = { size_t(fEnd - fStart) };
      // in case input tensor is not a dynamic tensor we should register the output as a Constant tensor since we know
      // its content
      if (!model.IsDynamicTensor(fNX) && !fOutput_shape.empty()) {
         std::shared_ptr<void> data(malloc(length * sizeof(int64_t)), free);
         auto shape_values = std::vector<int64_t>(fShape.begin()+fStart, fShape.begin() + fEnd );
         std::memcpy(data.get(), (void*) shape_values.data(), length * sizeof(int64_t));
         model.AddConstantTensor(fNY, ETensorType::INT64, fOutput_shape, data);
      }
      else
         model.AddIntermediateTensor(fNY, ETensorType::INT64, fOutput_shape);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Shape op called to Generate without being initialized first");
      }
      std::stringstream out;

      out << "\n//------ Shape\n";
      // add a dummy statement to avoid warning for unused input
      out << SP << "(void) tensor_" << fNX << ";\n";
      size_t length = ConvertShapeToLength(fOutput_shape);
      for (size_t id = 0; id < length; id++) {
         out << SP << "tensor_" << fNY << "["<< id << "] = " << fShape[fStart+id] << ";\n";
      }
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Shape
