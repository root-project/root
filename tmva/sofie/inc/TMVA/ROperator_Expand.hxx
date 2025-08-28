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

   std::vector<Dim> fShapeX;
   std::vector<size_t> fShape;
   std::vector<Dim> fShapeY;
   std::vector<Dim> fShapeDim;

   std::string fNX;
   std::string fNShape;
   std::string fNY;
   std::string fType;

   bool fInitialized = false;
   bool fInitializedShape = false;
   bool fInitBroadcast = false;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::string nameShape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNShape(UTILITY::Clean_name(nameShape)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }


   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
        throw std::runtime_error("TMVA SOFIE Expand Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.IsInitializedTensor(fNShape)) {
         fInitializedShape = true;
         int64_t *shapeData =
           static_cast<int64_t *>(model.GetInitializedTensorData(fNShape).get());
         fShape = model.GetTensorShape(fNShape);
         if (fShape.size() != 1) {
            throw std::runtime_error("TMVA::SOFIE - Expand operator shape must be a 1d tensor.");
         }
         size_t N = fShape[0];
         // what do we do if shapeData contains negative values?
         for (size_t i = 0; i < N; i++) {
            if ( shapeData[i] < 0)
               throw std::runtime_error("TMVA::SOFIE - Expand: invalid shape value " + std::to_string(shapeData[i]));
         }
         std::vector<size_t> shape(shapeData, shapeData + N);
         fShapeDim = ConvertShapeToDim(shape);
      } else if (model.IsShapeTensor(fNShape)) {
         // case input shape is a shape tensor
         fShapeDim = model.GetShapeTensorValues(fNShape);
         fInitializedShape = true;
      } else {
         // assume shape of input shape is known (size is 1)
         auto shapeOfInputShape = model.GetTensorShape(fNShape);
         fShapeDim.resize(shapeOfInputShape[0]);
         for (size_t i = 0; i < fShapeDim.size(); i++) {
            fShapeDim[i] = Dim{std::string("v_") + fNShape + "_" + std::to_string(i)};
            model.AddShapeParam(fShapeDim[i].param);
         }
      }
      // Y is the common shape of fShapeX and shape
      auto ret  = TMVA::Experimental::SOFIE::UTILITY::MultidirectionalBroadcastShape(fShapeX, fShapeDim);
      fShapeY = ret.second;
      fInitialized = model.IsInitializedTensor(fNX) && fInitializedShape;
      std::vector<size_t> shapeX;
      std::vector<size_t> shapeY;
      // case shape tensor and input shape are known
      if (!model.IsDynamicTensor(fNX) && !model.IsDimInputTensor(fNX) && fInitializedShape) {
         shapeX = ConvertShapeToInt(fShapeX);
         shapeY = ConvertShapeToInt(fShapeY);
         if (!UTILITY::AreSameShape(shapeX, shapeY))
            fInitBroadcast = true;
      }
      if (fInitialized) {
         // cannot have Dim initialized tensors
         assert(!shapeX.empty() && !shapeY.empty());
         // Broadcast X to the common shape shapeY
         // If X is an initialized tensor (constant)
         auto data = model.GetInitializedTensorData(fNX);
         if (fInitBroadcast) {
            std::shared_ptr<void> broadcastedData(
               UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), shapeX, shapeY),
               std::default_delete<T[]>());
            // Update the data and the shape of X
            model.UpdateInitializedTensor(fNX, model.GetTensorType(fNX), shapeY, broadcastedData);
            fShapeX = fShapeY;
            // need to set as a not writable tensor
            model.SetNotWritableInitializedTensor(fNX);
            data = broadcastedData;
         }
         if (fInitBroadcast || model.IsConstantTensor(fNX)) {
            fIsOutputConstant = true; // constant output in this case
            model.AddConstantTensor(fNY, model.GetTensorType(fNX), shapeY, data);
            fOutputTensorNames.pop_back();
         } else {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), shapeY);
         }
      } else {
         // // case input is not initialized
         // if (shapeX.empty() && shapeDim.empty()) {

         // }
         // if (fInitializedShape)
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      }
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Expand - input " << fNX << " shape " << ConvertShapeToString(fShapeX) << " --> " << fNY << " shape "
                  << ConvertShapeToString(fShapeY) << (fIsOutputConstant ? ConvertValuesToString(model.GetTensorData<T>(fNY)) + " (constant)" : "") << std::endl;
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      if (!fIsOutputConstant && fInitialized && !fInitBroadcast) {
         // shapeX and shapeY are the same in this case
         auto length = ConvertDimShapeToLength(fShapeY);
         out << "// Copying initialized tensor " << fNX << " to " << fNY << "\n";
         out << SP << "std::copy(tensor_" << fNX << ", " << "tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";
      }
      return out.str();
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Expand Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ Expand " << opName << " --> " << ConvertShapeToString(fShapeY) << "\n";
      // need to declare shape parameters for non initialized shapes
      if (!fInitializedShape) {
         for (size_t i = 0; i < fShapeDim.size(); i++) {
            out << SP << "size_t " << fShapeDim[i] << " = " << "tensor_" << fNShape << "[" << i << "];\n";
         }
      }
      // No need to broadcast A if it's an initialized tensor or shapes are the same
      if (!fInitialized && fShapeX != fShapeY) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX << "\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << fType << ">(tensor_" << fNX << ", " << ConvertShapeToString(fShapeX) << ", " << ConvertShapeToString(fShapeY)
                   << ", std::span<"<<fType<<">(tensor_"<<fNY<<", "<<ConvertDimShapeToLength(fShapeY)<<"));\n";
      }
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Expand
