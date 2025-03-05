#ifndef TMVA_SOFIE_ROPERATOR_Pad
#define TMVA_SOFIE_ROPERATOR_Pad

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Pad final : public ROperator
{
public:
   enum EMode { kConstant, kReflect, kEdge, kWrap };
private:

   std::string fNX;
   std::string fNP;
   std::string fNCV;
   std::string fNAX;
   std::string fNY;
   T fConstantValue;
   EMode fMode;
   std::vector<size_t> fInputShape;
   std::vector<size_t> fOutputShape;
   std::vector<std::pair<int64_t, int64_t>> fPads;

public:

   ROperator_Pad(){}
   ROperator_Pad(const std::string & nameX, const std::string & nameP,  const std::string & nameCV,
                 const std::string & nameAX, const std::string & nameY, const std::string & mode) :
      fNX(UTILITY::Clean_name(nameX)), fNP(UTILITY::Clean_name(nameP)),
      fNCV(UTILITY::Clean_name(nameCV)), fNAX(UTILITY::Clean_name(nameAX)),
      fNY(UTILITY::Clean_name(nameY))
      {
         fMode = kConstant;
         if (mode == "constant")
            fMode = kConstant;
         else if (mode == "reflect")
            fMode = kReflect;
         else if (mode == "edge")
            fMode = kEdge;
         else if (mode == "wrap")
            fMode = kWrap;
         
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Pad Op Input Tensor is not found in model");
      }

      fInputShape = model.GetTensorShape(fNX);

      if (fMode != EMode::kConstant) {
         throw std::runtime_error("TMVA SOFIE Pad Op supports now only Constant mode");
      }

      // get pads data
      int64_t * padsData = nullptr;
      if (model.IsInitializedTensor(fNP)) {
         padsData = static_cast<int64_t*>(model.GetInitializedTensorData(fNP).get());
      } else {
         throw std::runtime_error("TMVA SOFIE Pad Op supports now only initialized Pads data");
      }
      // get constant value
      fConstantValue = 0;
      if (!fNCV.empty()) {
         if (model.IsInitializedTensor(fNCV)) {
            T * cData = static_cast<T*>(model.GetInitializedTensorData(fNCV).get());
            fConstantValue = cData[0];
         } else {
            throw std::runtime_error("TMVA SOFIE Pad Op supports now only initialized Constant Value  data");
         }
      }
      std::vector<int64_t> axes;
      if (!fNAX.empty()) {
         if (model.IsInitializedTensor(fNAX)) {
            auto shape = model.GetTensorShape(fNAX);
            // it should be a 1D tensor
            size_t nax = shape[0];
            // switch types
            if (model.GetTensorType(fNAX) == ETensorType::INT64) {
               auto data = static_cast<int64_t*>(model.GetInitializedTensorData(fNAX).get());
               axes = std::vector<int64_t>(data, data + nax);
            } else if (model.GetTensorType(fNAX) == ETensorType::INT32) {
               auto data = static_cast<int32_t*>(model.GetInitializedTensorData(fNAX).get());
               axes.resize(nax);
               for (size_t i = 0; i < nax; i++)
                  axes[i] = data[i];
            }  else {
               throw std::runtime_error("TMVA SOFIE Pad Op invalid input Axes type");
            }
         } else {
            throw std::runtime_error("TMVA SOFIE Pad Op supports now only initialized Axes data");
         }
      }


      fOutputShape = fInputShape;
      size_t axesSize = axes.size();
      if (axesSize == 0) {
         for (size_t i = 0; i < fInputShape.size(); i++) {
            axes.push_back(i);
         }
         axesSize = fInputShape.size();
      }
      fPads.resize(fInputShape.size());
      for (size_t i = 0; i < fInputShape.size(); i++) {
         if (axes[i] < 0) axes[i] += fInputShape.size();
         if (axes[i] == int64_t(i)) {
            fPads[i].first = padsData[i];
            fPads[i].second = padsData[axesSize + i];
            int64_t outDim = static_cast<int64_t>(fOutputShape[i]) + fPads[i].first + fPads[i].second;
            if (outDim < 0)
               throw std::runtime_error("TMVA SOFIE Pad Op : invalid Pads values");
            fOutputShape[i] = outDim;
         }
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fOutputShape);

      if (model.Verbose()) {
         std::cout << "initializing Pad operator with pads ..  : ";
         for (auto & p : fPads)
            std::cout << "{ " << p.first << " , " << p.second << "} ";
         std::cout << std::endl;
         std::cout <<  "Pad: " << fNX << " " << ConvertShapeToString(fInputShape) << " -> " << fNY << " with shape " << ConvertShapeToString(fOutputShape)
                  << std::endl;
      }

   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fOutputShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Pad called to Generate without being initialized first");
      }
      std::stringstream out;
      auto inputStride = UTILITY::ComputeStrideFromShape(fInputShape);
      auto outStride = UTILITY::ComputeStrideFromShape(fOutputShape);
      out << "\n//------ Pad\n";
      // fill first output tensor with the constant values
      int length = ConvertShapeToLength(fOutputShape);
      int dims = fOutputShape.size();
      out << "std::fill(tensor_" << fNY << ", tensor_" << fNY << " + " << length << ","
          << fConstantValue << ");\n";

      // copy now data from input tensor in output ones
      for (int i = 0; i < dims; i++) {
         for (int j = 1; j < i; j++) out << SP;
         out << "for (int id" << i << " = 0; id" << i << " < " << fInputShape[i] << "; id"
             << i << "++) {\n";
      }
      // compute index from strides
      //linear_index = i_1 * stride[0] + i_2 * stride[1] + ... + i_N * stride[N-1]
      for (int j = 0; j < dims; j++) out << SP;
      out << "tensor_" << fNY << "[";
      for (int i = 0; i < dims; i++) {
         out << "(id" << i;
         if (fPads[i].first != 0) out << " + " << fPads[i].first;
         out << ")";
         if (i < dims-1) out << " * " << outStride[i] << " + ";
      }
      out << "] =\n     tensor_" << fNX << "[";
      for (int i = 0; i < dims; i++) {
         out << "id" << i;
         if (i < dims-1) out << " * " << inputStride[i] << " + ";
      }
      out << "];\n";
      for (int i = dims-1; i >= 0; i--) {
         for (int j = 1; j < i; j++) out << SP;
         out << "}\n";
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Swish
