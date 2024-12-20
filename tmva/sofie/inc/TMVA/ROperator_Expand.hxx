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
   std::string fNShape;
   std::string fNY;
   std::string fType;

   bool fInitialized = false;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::string nameShape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNShape(UTILITY::Clean_name(nameShape)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
        throw std::runtime_error("TMVA SOFIE Expand Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (!model.IsInitializedTensor(fNShape)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNShape + " is not initialized.");
      }
      int64_t *shapeData =
           static_cast<int64_t *>(model.GetInitializedTensorData(fNShape).get());
      fShape = model.GetTensorShape(fNShape);
      if (fShape.size() != 1) {
         throw std::runtime_error("TMVA::SOFIE - Expand operator shape must be a 1d tensor.");
      }
      size_t N = fShape[0];
      std::vector<size_t> shape(shapeData, shapeData + N);
      // Y is the common shape of fShapeX and shape
      fShapeY = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcastShape(
        fShapeX, shape);
      fInitialized = model.IsInitializedTensor(fNX);
      // Broadcast X to the common shape fShapeY
      bool broadcast = !UTILITY::AreSameShape(fShapeX, fShapeY);
      if (broadcast && model.IsInitializedTensor(fNX)) {
         // If X is an initialized tensor (constant)
         auto data = model.GetInitializedTensorData(fNX);
         std::shared_ptr<void> broadcastedData(
               UTILITY::UnidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeX, fShapeY),
               std::default_delete<float[]>());
         // Update the data and the shape of X
         model.UpdateInitializedTensor(fNX, model.GetTensorType(fNX), fShapeY, broadcastedData);
         fShapeX = fShapeY;
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose())
         std::cout << "Expand - output is with shape " << ConvertShapeToString(fShapeY) << std::endl;
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
      out << SP << "\n//------ Expand Op" << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      // No need to broadcast A if it's an initialized tensor
      if (fInitialized) {
         out << "// Copying initialized tensor " << fNX << " to " << fNY << "\n";
         out << SP << "std::copy(tensor_" << fNX << ", " << "tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";
      } else {
         out << SP << "// Broadcasting uninitialized tensor " << fNX << "\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_" << fNX << ", " << ConvertShapeToString(fShapeX) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNY << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Expand
