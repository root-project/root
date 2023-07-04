
#ifndef TMVA_SOFIE_ROperator_Equal
#define TMVA_SOFIE_ROperator_Equal

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template<typename T>
class ROperator_Equal final : public ROperator{
private:

   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShapeX1;
   std::vector<size_t> fShapeX2;
   std::vector<size_t> fShapeY;
   std::string fNBroadcastedX1;
   std::string fNBroadcastedX2;
   bool fBroadcast = false;


public:
   ROperator_Equal(){}
   ROperator_Equal(std::string nameX1, std::string nameX2, std::string nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX1)){
         throw std::runtime_error(std::string("TMVA SOFIE Equal Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNX2)) {
         throw std::runtime_error(std::string("TMVA SOFIE Equal Op Input Tensor ") + fNX2 + "is not found in model");
      }
      fShapeX1 = model.GetTensorShape(fNX1);
      fShapeX2 = model.GetTensorShape(fNX2);
      bool broadcast = !UTILITY::AreSameShape(fShapeX1, fShapeX2);
      if (broadcast) {
         // Y is the common shape of A and B
         fShapeY = UTILITY::UnidirectionalBroadcastShape(fShapeX1, fShapeX2);
         bool broadcastX1 = !UTILITY::AreSameShape(fShapeX1, fShapeY);
         bool broadcastX2 = !UTILITY::AreSameShape(fShapeX2, fShapeY);
         // Broadcast A to Y
         if (broadcastX1) {
            if (model.IsInitializedTensor(fNX1)) {
               auto data = model.GetInitializedTensorData(fNX1);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeX1, fShapeY),
                  std::default_delete<float[]>());
               // Update the data and the shape of A
               model.UpdateInitializedTensor(fNX1, model.GetTensorType(fNX1), fShapeY, broadcastedData);
               fShapeX1 = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting A
               fNBroadcastedX1 = "Broadcasted" + fNX1;
               model.AddIntermediateTensor(fNBroadcastedX1, model.GetTensorType(fNX1), fShapeY);
            }
         }
         // Broadcast B to Y
         if (broadcastX2) {
            if (model.IsInitializedTensor(fNX2)) {
               auto data = model.GetInitializedTensorData(fNX2);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeX2, fShapeY),
                  std::default_delete<float[]>());
               // Update the data and the shape of B
               model.UpdateInitializedTensor(fNX2, model.GetTensorType(fNX2), fShapeY, broadcastedData);
               fShapeX2 = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting B
               fNBroadcastedX2 = "Broadcasted" + fNX2;
               model.AddIntermediateTensor(fNBroadcastedX2, model.GetTensorType(fNX2), fShapeY);
            }
         }
      } else {
         fShapeY = fShapeX1;
      }
      model.AddIntermediateTensor(fNY, ETensorType::BOOL , fShapeY);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;

     if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Equal Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ Equal\n";
      size_t length = ConvertShapeToLength(fShapeY);
      // Broadcast A if it's uninitialized
      if (!fNBroadcastedX1.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX1 << "\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_" << fNX1 << ", " << ConvertShapeToString(fShapeX1) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX1 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcastedX2.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX2 << "\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_" << fNX2 << ", " << ConvertShapeToString(fShapeX2) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX2 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      const std::string& nameX1 = fNBroadcastedX1.empty()? fNX1 : fNBroadcastedX1;
      const std::string& nameX2 = fNBroadcastedX2.empty()? fNX2 : fNBroadcastedX2;
     
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (tensor_" << nameX1 << "[id] == tensor_" << nameX2 <<"[id]) ? true : false;\n";
      out << SP << "}\n";
   
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Equal