
#ifndef TMVA_SOFIE_ROperator_Comparision
#define TMVA_SOFIE_ROperator_Comparision

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EComparisionOperator { Eq, Less, LessEq, Greater, GreaterEq };

template <typename T, EComparisionOperator Op1>
struct ComparisionTrait{};

template <typename T>
struct ComparisionTrait<T, Eq> {
   static const std::string Name() { return "Equal"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " == " +  t2; }
   static bool Result(T v1, T v2) { return v1 == v2;}
};

template <typename T>
struct ComparisionTrait<T, Less> {
   static const std::string Name() { return "Less"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " < " + t2; }
   static bool Result(T v1, T v2) { return v1 < v2;}
};

template <typename T>
struct ComparisionTrait<T, LessEq> {
   static const std::string Name() { return "LessOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " <= " +  t2;  }
   static bool Result(T v1, T v2) { return v1 <= v2;}
};

template <typename T>
struct ComparisionTrait<T, Greater> {
   static const std::string Name() { return "Greater"; }
   static std::string Op(const std::string & t1, const std::string t2) { return  t1 + " > " +  t2; }
   static bool Result(T v1, T v2) { return v1 > v2;}
};

template <typename T>
struct ComparisionTrait<T, GreaterEq> {
   static const std::string Name() { return "GreaterOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " >= " +  t2 ; }
   static bool Result(T v1, T v2) { return v1 >= v2;}
};

template<typename T, EComparisionOperator Op>
class ROperator_Comparision final : public ROperator{
private:

   bool fIsModelOutput = false;
   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShapeX1;
   std::vector<size_t> fShapeX2;
   std::vector<Dim> fDimShapeX1;
   std::vector<Dim> fDimShapeX2;
   std::vector<size_t> fShapeY;
   std::string fNBroadcastedX1;
   std::string fNBroadcastedX2;
   ETensorType fTensorType1 = ETensorType::UNDEFINED;
   ETensorType fTensorType2 = ETensorType::UNDEFINED;
   bool fBroadcast = false;


public:
   ROperator_Comparision(){}
   ROperator_Comparision(const std::string & nameX1, const std::string & nameX2, const std::string & nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX1, fNX2 };
         
         // output will be a boolean vector so should not be considered for memory optimized pool
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX1)){
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNX2)) {
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX2 + "is not found in model");
      }
      if (model.IsDynamicTensor(fNX1))
         fDimShapeX1 = model.GetDynamicTensorShape(fNX1);
      else {
         fShapeX1 = model.GetTensorShape(fNX1);
         fDimShapeX1 = ConvertShapeToDim(fShapeX1);
      }
      if (model.IsDynamicTensor(fNX2))
         fDimShapeX2 = model.GetDynamicTensorShape(fNX2);
      else {
         fShapeX2 = model.GetTensorShape(fNX2);
         fDimShapeX2 = ConvertShapeToDim(fShapeX2);
      }
      fShapeX1 = model.GetTensorShape(fNX1);
      fShapeX2 = model.GetTensorShape(fNX2);
      fTensorType1 = model.GetTensorType(fNX1);
      fTensorType2 = model.GetTensorType(fNX2);
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
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX1, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of A
               model.UpdateInitializedTensor(fNX1, model.GetTensorType(fNX1), fShapeY, broadcastedData);
               fShapeX1 = fShapeY;
            }
         }
         // Broadcast B to Y
         if (broadcastX2) {
            if (model.IsInitializedTensor(fNX2)) {
               auto data = model.GetInitializedTensorData(fNX2);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX2, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of B
               model.UpdateInitializedTensor(fNX2, model.GetTensorType(fNX2), fShapeY, broadcastedData);
               fShapeX2 = fShapeY;
            }
         }
      } else {
         fShapeY = fShapeX1;
      }
      // case of constant tensors
      if (model.IsInitializedTensor(fNX1) && model.IsInitializedTensor(fNX2) ) {
         fIsOutputConstant = true;
         auto data1 = static_cast<T *>(model.GetInitializedTensorData(fNX1).get());
         auto data2 = static_cast<T *>(model.GetInitializedTensorData(fNX2).get());
         size_t length = ConvertShapeToLength(fShapeY);
         bool * outData = new bool[length];
         for (size_t i = 0; i < length; i++)
            outData[i] = ComparisionTrait<T,Op>::Result(data1[i], data2[i]);
         model.AddConstantTensor(fNY, fShapeY, outData);
         if (model.Verbose())
            std::cout <<  ComparisionTrait<T,Op>::Name() << " op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
               << ConvertValuesToString(length,outData) << std::endl;
         delete [] outData;
      } else {
         model.AddIntermediateTensor(fNY, ETensorType::BOOL , fShapeY);
      }
      // check if this is not output operators to add a specific line for definining the tensor_xxx variable
      const auto & outputTensorNames = model.GetOutputTensorNames();
      fIsModelOutput = false;
      if (std::find(outputTensorNames.begin(), outputTensorNames.end(), fNY) != outputTensorNames.end())
         fIsModelOutput = true;
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << ComparisionTrait<T,Op>::Name() << "  " << opName
                                 << " --> " << ConvertShapeToString(fShapeY) << "\n";

      auto stridesX1 = UTILITY::ComputeStrideFromShape(fShapeX1);
      auto stridesX2 = UTILITY::ComputeStrideFromShape(fShapeX2);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);

      std::string compute_idx_X1, compute_idx_X2, compute_idx_Y;
      if (std::all_of(fShapeX1.begin(), fShapeX1.end(), [](size_t x) { return x == 1; })){
         compute_idx_X1 = "0";
      } else {
         for(size_t i = 0; i<fShapeX1.size(); ++i){
            if(fShapeX1[i]==1) continue;
            compute_idx_X1 += " idx_"+fNY+std::to_string(i+(fShapeY.size()-fShapeX1.size()))+" * "+std::to_string(stridesX1[i])+" +";
         }
         compute_idx_X1.pop_back();
      }
      if (std::all_of(fShapeX2.begin(), fShapeX2.end(), [](size_t x) { return x == 1; })){
         compute_idx_X2 = "0";
      } else {
         for(size_t i = 0; i<fShapeX2.size(); ++i){
            if(fShapeX2[i]==1) continue;
            compute_idx_X2 += " idx_"+fNY+std::to_string(i+(fShapeY.size()-fShapeX2.size()))+" * "+std::to_string(stridesX2[i])+" +";
         }
         compute_idx_X2.pop_back();
      }
      
      for(size_t i=0; i<fShapeY.size(); ++i){
         if(fShapeY[i]!=1){
            out<<std::string(i + 1, ' ')<<"for(size_t idx_"<<fNY<<i<<"=0; idx_"<<fNY<<i<<"<"<<fShapeY[i]<<"; ++idx_"<<fNY<<i<<"){\n";
            compute_idx_Y += "idx_"+fNY+std::to_string(i)+"*"+std::to_string(stridesY[i])+"+";
         }
      }
      compute_idx_Y.pop_back();
      out << SP << SP << "tensor_" << fNY <<"["<<compute_idx_Y<<"] = "<<ComparisionTrait<T,Op>::Op("tensor_"+ fNX1 + "["+compute_idx_X1+"]", "tensor_"+ fNX2 + "["+compute_idx_X2+"]")<<" ;\n";
      for(size_t i=0; i<fShapeY.size(); ++i){
         if(fShapeY[i]!=1){
            out<<std::string(fShapeY.size()-i+1, ' ')<<"}\n";
         }
      }

      // since output is a boolean need to add the tensor_xxx variable since it is not defined as a pointer to a boolean std::vector
      if (!fIsModelOutput)
         out << SP << "const std::vector<std::uint8_t> & tensor_" << fNY << " = fTensor_" << fNY << ";\n";

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Comparision
