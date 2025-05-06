#ifndef TMVA_SOFIE_ROperator_Where
#define TMVA_SOFIE_ROperator_Where

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{



template<typename T>
class ROperator_Where final : public ROperator{
private:

   bool fIsInputBoolTensor = false;


   std::string fNA;
   std::string fNB;
   std::string fNC;
   std::string fNBroadcastedA;
   std::string fNBroadcastedB;
   std::string fNBroadcastedC;
   std::string fNY;


   std::vector<size_t> fShapeA;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeC;
   std::vector<size_t> fShapeY;


public:
   ROperator_Where(){}
   ROperator_Where(const std::string & nameA, const std::string & nameB, const std::string & nameC, const std::string & nameY):
      fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNA, fNB, fNC };
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNA)){
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNA + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNB + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNC)) {
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNC + "is not found in model");
      }
      // check if fNC input tensor is boolean
      if (model.IsReadyInputTensor(fNC))
         fIsInputBoolTensor = true;
      // check broadcast for A, B and C
      fShapeA = model.GetTensorShape(fNA);
      fShapeB = model.GetTensorShape(fNB);
      fShapeC = model.GetTensorShape(fNC);
      bool broadcast = !UTILITY::AreSameShape(fShapeA, fShapeB) || !UTILITY::AreSameShape(fShapeA, fShapeC);
      if (broadcast) {
         // find shape to broadcast between A,B,C looking for max length
         size_t lengthA = ConvertShapeToLength(fShapeA);
         size_t lengthB = ConvertShapeToLength(fShapeB);
         size_t lengthC = ConvertShapeToLength(fShapeC);
         bool broadcastA = false, broadcastB = false, broadcastC = false;
         if (lengthA >= lengthB && lengthA >= lengthC) {
            fShapeY = fShapeA;
            //broadcast B and C if different than A
            broadcastB = (lengthB != lengthA);
            broadcastC = (lengthC != lengthA);
         }
         else if (lengthB >= lengthA && lengthB >= lengthC) {
            fShapeY = fShapeB;
            //broadcast A and C if different than B
            broadcastA = (lengthA != lengthB);
            broadcastC = (lengthC != lengthB);
         }
         else if (lengthC >= lengthA && lengthC >= lengthB) {
            fShapeY = fShapeC;
            //broadcast A and B if different than C
            broadcastA = (lengthA != lengthC);
            broadcastB = (lengthB != lengthC);
         }

         // Broadcast A to Y
         if (broadcastA) {
            fNBroadcastedA = "BC_" + fNA + "_to_" + fNY;
            if (model.IsInitializedTensor(fNA)) {
               auto data = model.GetInitializedTensorData(fNA);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeA, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of A
               model.AddConstantTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY, broadcastedData);
               fShapeA = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting A
               model.AddIntermediateTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY);
            }
         }
         // Broadcast B to Y
         if (broadcastB) {
            fNBroadcastedB = "BC_" + fNB + "_to_" + fNY;
            if (model.IsInitializedTensor(fNB)) {
               auto data = model.GetInitializedTensorData(fNB);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeB, fShapeY),
                  std::default_delete<T[]>());
               // do not update tensor B but add broadcasted one (since it can be input to some other operators)
               model.AddConstantTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY, broadcastedData);
               fShapeB = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting B
               model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY);
            }
         }
         // Broadcast C to Y
         if (broadcastC) {
            fNBroadcastedC = "BC_" + fNC + "_to_" + fNY;
            if (model.IsInitializedTensor(fNC)) {
               auto data = model.GetInitializedTensorData(fNC);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeC, fShapeY),
                  std::default_delete<T[]>());
               // do not update tensor C but add broadcasted one (since it can be input to some other operators)
               model.AddConstantTensor(fNBroadcastedC, model.GetTensorType(fNC), fShapeY, broadcastedData);
               fShapeC = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting B
               model.AddIntermediateTensor(fNBroadcastedC, model.GetTensorType(fNC), fShapeY);
            }
         }
      } else {
         fShapeY = fShapeA;
      }
      // check case of constant  output (if all inputs are defined)
      if (model.IsInitializedTensor(fNC)) {

         std::string nameC = fNBroadcastedC.empty()? fNC : fNBroadcastedC;
         auto dataC = static_cast<bool *>(model.GetInitializedTensorData(nameC).get());
         model.SetNotWritableInitializedTensor(nameC);
         T * dataA = nullptr;
         T * dataB = nullptr;
         std::vector<Dim> shapeDataA;
         std::vector<Dim> shapeDataB;
         if (model.IsInitializedTensor(fNA)) {
             std::string nameA = fNBroadcastedA.empty()? fNA : fNBroadcastedA;
             dataA = static_cast<T *>(model.GetInitializedTensorData(nameA).get());
            // flag tensors to not be written in a file
            model.SetNotWritableInitializedTensor(nameA);
         } else if (model.IsShapeTensor(fNA))
            shapeDataA = model.GetShapeTensorValues(fNA);
         if (model.IsInitializedTensor(fNB)) {
            std::string nameB = fNBroadcastedB.empty()? fNB : fNBroadcastedB;
            dataB = static_cast<T *>(model.GetInitializedTensorData(nameB).get());
            model.SetNotWritableInitializedTensor(nameB);
         } else if (model.IsShapeTensor(fNB))
            shapeDataB = model.GetShapeTensorValues(fNB);

         std::vector<T> dataY;
         std::vector<Dim> shapeDataY;

         bool isOutputConstantTensor = true;
         if (dataA && dataB) {
            dataY.resize(ConvertShapeToLength(fShapeY));
            for (size_t i = 0; i < dataY.size(); i++)
                dataY[i] = (dataC[i]) ? dataA[i] : dataB[i];
         }
         else if (dataA && shapeDataB.size()>0 ) {
            shapeDataY.resize(ConvertShapeToLength(fShapeY));
            for (size_t i = 0; i < shapeDataY.size(); i++) {
               shapeDataY[i] = (dataC[i]) ? Dim{size_t(dataA[i])} : shapeDataB[i];
               isOutputConstantTensor &= !shapeDataY[i].isParam;
            }
         }
         else if (dataB && shapeDataA.size()>0 ) {
            shapeDataY.resize(ConvertShapeToLength(fShapeY));
            for (size_t i = 0; i < shapeDataY.size(); i++) {
               shapeDataY[i] = (dataC[i]) ? shapeDataB[i] : Dim{size_t(dataB[i])};
               isOutputConstantTensor &= !shapeDataY[i].isParam;
            }
         }
         else if (shapeDataB.size() > 0  && shapeDataA.size()>0 ) {
            shapeDataY.resize(ConvertShapeToLength(fShapeY));
            for (size_t i = 0; i < shapeDataY.size(); i++) {
               shapeDataY[i] = (dataC[i]) ? shapeDataA[i] : shapeDataB[i];
               isOutputConstantTensor &= !shapeDataY[i].isParam;
            }
         }
         fIsOutputConstant = true;  // this contains both case constant tensor output ans shape tensor output
         if (isOutputConstantTensor && dataY.empty()) {
            dataY.resize(shapeDataY.size());
            for (size_t i = 0; i < shapeDataY.size(); i++)
               dataY[i] = static_cast<T>(shapeDataY[i].dim);
         }
         if (dataY.size() > 0)
            model.AddConstantTensor<T>(fNY, fShapeY, dataY.data());
         else if (shapeDataY.size() > 0 )
           model.AddShapeTensor(fNY, shapeDataY, fShapeY.size() == 0);
         else {
            fIsOutputConstant = false;
         }
         if (fIsOutputConstant && model.Verbose())
            std::cout << "Where op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
               << ((dataY.size() > 0) ? ConvertValuesToString(dataY) : ConvertShapeToString(shapeDataY) )
               << ((dataY.size() > 0) ? " (constant)" : " (shape)") << std::endl;

         // output is a constant tensor
         if (fIsOutputConstant) fOutputTensorNames.pop_back();
      }
      if (!fIsOutputConstant) {
        model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
        if (model.Verbose())
            std::cout << "Where op " << " condition : " << fNC << "  " << ConvertShapeToString(fShapeC) <<
                   " X " << fNA << "  " << ConvertShapeToString(fShapeA) << " Y " <<  fNB << "  " << ConvertShapeToString(fShapeB)
                   << " ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << std::endl;
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override {

      if (fIsOutputConstant) return "";

      opName = "op_" + opName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Where Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//-------- Where " << opName << " --> " << ConvertShapeToString(fShapeY) << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      std::string typeName = TensorType<T>::Name();
      // Broadcast A if it's uninitialized
      if (fShapeA != fShapeY) {
         out << SP << "// Broadcasting uninitialized tensor " << fNA << "\n";
         //out << SP << "{\n";
         out << SP  << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNA << ", " << ConvertShapeToString(fShapeA) << ", " << ConvertShapeToString(fShapeY)
                         << ", fTensor_" << fNBroadcastedA << ");\n";
      }
      // Broadcast B if it's uninitialized
      if (fShapeB != fShapeY) {
         out << SP << "// Broadcasting uninitialized tensor " << fNB << "\n";
         //out << SP << "{\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNB << ", " << ConvertShapeToString(fShapeB) << ", " << ConvertShapeToString(fShapeY)
                   << ", fTensor_" << fNBroadcastedB << ");\n";
      }
       // Broadcast C if it's uninitialized
      if (fShapeC != fShapeY) {
         // special case if C is an input tensor
         if (fIsInputBoolTensor) {
            size_t inputLength = ConvertShapeToLength(fShapeC);
            out << SP << "std::vector<std::uint8_t> fTensor_" << fNC << "(tensor_" << fNC <<  ", tensor_" << fNC << " + " << inputLength << ");\n";
         }
         out << SP << "// Broadcasting uninitialized tensor " << fNC << "\n";
         //out << SP << "{\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<std::uint8_t>(fTensor_" << fNC << ".data(), " << ConvertShapeToString(fShapeC) << ", " << ConvertShapeToString(fShapeY)
                   << ", fTensor_" << fNBroadcastedC << ");\n";
      }
      std::string nameA = fNBroadcastedA.empty()? fNA : fNBroadcastedA;
      std::string nameB = fNBroadcastedB.empty()? fNB : fNBroadcastedB;
      std::string nameC = fNBroadcastedC.empty()? fNC : fNBroadcastedC;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      // get output tensor applying condition
      out << SP << SP << "tensor_" << fNY << "[id] = "  << "(fTensor_" << nameC << "[id]) ? tensor_"
                               << nameA << "[id] : tensor_" + nameB + "[id];\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Where
