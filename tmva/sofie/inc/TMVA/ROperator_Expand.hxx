#ifndef TMVA_SOFIE_ROperator_Expand
#define TMVA_SOFIE_ROperator_Expand

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template<typename T>
class ROperator_Expand final : public ROperator{
private:

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShape;
   std::vector<size_t> fShapeY;

   std::string fNX;
   std::string fNY;
   std::string fNShape;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::string nameShape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNShape(UTILITY::Clean_name(nameShape)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      std::vector<std::vector<size_t>> ret;
      for(auto it:fShape)
        ret[0].push_back(it);
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Expand Op Input Tensor is not found in model");
      }

      fShapeX = model.GetTensorShape(fNX);

      if(model.IsInitializedTensor(fNShape)){
            auto data = model.GetInitializedTensorData(fNShape);
            auto output_shape = static_cast<int64_t *>(data.get());
            auto vec = model.GetTensorShape(fNShape);
            assert(vec.size() == 1);
            size_t n = vec[0]; // size of shape input tensor
            std::copy(output_shape, output_shape + n, fShape.begin());
      }

      bool broadcast = !UTILITY::AreSameShape(fShapeX, fShape);
      if (broadcast) {
         // Y is the common shape of A and B
         fShapeY = fShape;
         bool broadcastX = !UTILITY::AreSameShape(fShapeX, fShapeY);
         
         // Broadcast A to Y
         if (broadcastX) {
            if (model.IsInitializedTensor(fNX)) {
               auto data = model.GetInitializedTensorData(fNX);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeX, fShapeY),
                  std::default_delete<float[]>());
               // Update the data and the shape of X
               model.UpdateInitializedTensor(fNX, model.GetTensorType(fNX), fShapeY, broadcastedData);
               fShapeX = fShapeY;
            } 
         }
      } 
      else {
         fShapeY = fShapeX;
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Expand Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ EXPAND" << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      // Broadcast A if it's uninitialized
      if (!fNY.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX << "\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_" << fNX << ", " << ConvertShapeToString(fShapeX) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNY << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
    
      const std::string& nameX = fNY.empty()? fNX : fNY;
      
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = tensor_" + fNX + "[id]" <<  " ;\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_BasicBinary