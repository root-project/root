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
      if (model.IsInitializedTensor(fNA) && model.IsInitializedTensor(fNB) && model.IsInitializedTensor(fNC)) {
         std::string nameA = fNBroadcastedA.empty()? fNA : fNBroadcastedA;
         std::string nameB = fNBroadcastedB.empty()? fNB : fNBroadcastedB;
         std::string nameC = fNBroadcastedC.empty()? fNC : fNBroadcastedC;
         auto dataA = static_cast<T *>(model.GetInitializedTensorData(nameA).get());
         auto dataB = static_cast<T *>(model.GetInitializedTensorData(nameB).get());
         auto dataC = static_cast<bool *>(model.GetInitializedTensorData(nameC).get());
         std::vector<T> dataY(ConvertShapeToLength(fShapeY));
         for (size_t i = 0; i < dataY.size(); i++)
             dataY[i] = (dataC[i]) ? dataA[i] : dataB[i];
         model.AddConstantTensor<T>(fNY, fShapeY, dataY.data());
         // flag tensors to not be written in a file
         model.SetNotWritableInitializedTensor(nameA);
         model.SetNotWritableInitializedTensor(nameB);
         model.SetNotWritableInitializedTensor(nameC);

         fIsOutputConstant = true;
         if (model.Verbose())
            std::cout << "Where op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
               << ConvertValuesToString(dataY) << std::endl;
         
         // output is a constant tensor
         fOutputTensorNames.pop_back();
      }
      else {
        model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string OpName) override {

      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Where Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//-------- Where   \n";
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
            out << SP << "std::vector<bool> fTensor_" << fNC << "(tensor_" << fNC <<  ", tensor_" << fNC << " + " << inputLength << ");\n";
         }
         out << SP << "// Broadcasting uninitialized tensor " << fNC << "\n";
         //out << SP << "{\n";
         // for boolean we need to pass vector<bool> and use the non-template version of the function
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast(fTensor_" << fNC << ", " << ConvertShapeToString(fShapeC) << ", " << ConvertShapeToString(fShapeY)
                   << ", fTensor_" << fNBroadcastedC << ");\n";
      }
      std::string nameA = fNBroadcastedA.empty()? fNA : fNBroadcastedA;
      std::string nameB = fNBroadcastedB.empty()? fNB : fNBroadcastedB;
      std::string nameC = fNBroadcastedC.empty()? fNC : fNBroadcastedC;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      // get output tensor applying condition (note we need to use directly the vector<bool> since v.data(),  i.e the data pointer, does not exist)
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
