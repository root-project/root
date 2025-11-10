
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
   std::vector<Dim> fDimShapeY;
   ETensorType fTensorType1 = ETensorType::UNDEFINED;
   ETensorType fTensorType2 = ETensorType::UNDEFINED;
   int fBroadcastFlag = 0;


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
      fTensorType1 = model.GetTensorType(fNX1);
      fTensorType2 = model.GetTensorType(fNX2);
      // case of non dynamic tensors
      if (!fShapeX1.empty() && !fShapeX2.empty()) {
         bool broadcastX1 = false;
         bool broadcastX2 = false;
         if (UTILITY::AreSameShape(fShapeX1, fShapeX2)) {
            // no broadcast needed
            fShapeY = fShapeX1;
         } else  {
            // Y is the common shape of A and B
            fShapeY = UTILITY::UnidirectionalBroadcastShape(fShapeX1, fShapeX2);
            broadcastX1 = !UTILITY::AreSameShape(fShapeX1, fShapeY);
            broadcastX2 = !UTILITY::AreSameShape(fShapeX2, fShapeY);
         }


         // analyze case of constant tensors or shape tensors (which have known shapes but data as Dim values
         // normal case with non-dynamic tensor is also here
         T *data1 = nullptr;
         T *data2 = nullptr;
         std::unique_ptr<T> broadcastedData1;
         std::unique_ptr<T> broadcastedData2;
         // data for shape tensors
         std::vector<Dim> shapeData1;
         std::vector<Dim> shapeData2;
         size_t length = ConvertShapeToLength(fShapeY);
         bool *outData = new bool[length];
         if (model.IsInitializedTensor(fNX1)) {
            data1 = static_cast<T *>(model.GetInitializedTensorData(fNX1).get());
            if (broadcastX1) {
               broadcastedData1 = std::unique_ptr<T>(
                  UTILITY::UnidirectionalBroadcast<T>(data1, fShapeX1, fShapeY));
               data1 = broadcastedData1.get();
            }

         } else if (model.IsShapeTensor(fNX1)) {
            shapeData1 = model.GetShapeTensorValues(fNX1);
         }
         if (model.IsInitializedTensor(fNX2)) {
            data2 = static_cast<T *>(model.GetInitializedTensorData(fNX2).get());
            if (broadcastX2) {
               broadcastedData2 = std::unique_ptr<T>(
                  UTILITY::UnidirectionalBroadcast<T>(data2, fShapeX2, fShapeY));
               data2 = broadcastedData2.get();
            }
         } else if (model.IsShapeTensor(fNX2)) {
            shapeData2 = model.GetShapeTensorValues(fNX2);
         }
         if (data1 && data2) {
            fIsOutputConstant = true;
            for (size_t i = 0; i < length; i++)
               outData[i] = ComparisionTrait<T, Op>::Result(data1[i], data2[i]);
            model.AddConstantTensor(fNY, fShapeY, outData);
            if (model.Verbose())
               std::cout << ComparisionTrait<T, Op>::Name() << " op ---> " << fNY << "  "
                         << ConvertShapeToString(fShapeY) << " : " << ConvertValuesToString(length, outData)
                         << std::endl;
         } else if ((data1 || !shapeData1.empty()) && (data2 || !shapeData2.empty())) {
            fIsOutputConstant = true;
            if (data1 && !data2) {
               // data 1 is constant and data2 is shape
               for (size_t i = 0; i < length; i++) {
                  if (shapeData2[i].isParam) {
                     if (shapeData2[i].dim == size_t(-1) || data1[i] > 0) {
                        fIsOutputConstant = false;
                        break;
                     } else {
                        // assume a comparison is done with .dim = 0
                        shapeData2[i].dim = 0;
                     }
                  }
                  outData[i] = ComparisionTrait<T, Op>::Result(data1[i], static_cast<T>(shapeData2[i].dim));
               }
            } else if (!data1 && data2) {
               // data 1 is shape and dat2 is constant
               for (size_t i = 0; i < length; i++) {
                  if (shapeData1[i].isParam) {
                     if (shapeData1[i].dim == size_t(-1) || data2[i] > 0) {
                        fIsOutputConstant = false;
                        break;
                     } else {
                        // assume a comparison is done with .dim = 0
                        shapeData1[i].dim = 0;
                     }
                  }
                  outData[i] = ComparisionTrait<T, Op>::Result(static_cast<T>(shapeData1[i].dim), data2[i]);
               }
            } else if (!shapeData1.empty() && !shapeData2.empty()) {
               // both data1 and data2 are shape tensors
               for (size_t i = 0; i < length; i++) {
                  if (!shapeData1[i].isParam && !shapeData2[i].isParam) {
                     outData[i] = ComparisionTrait<T, Op>::Result(shapeData1[i].dim, shapeData2[i].dim);
                  } else if (shapeData1[i].isParam && shapeData2[i].isParam) {
                     if (shapeData1[i].param == shapeData2[i].param)
                        outData[i] = ComparisionTrait<int, Op>::Result(1, 1); // comparison of two equal value
                     else {
                        fIsOutputConstant = false;
                        break;
                     }
                  } else {
                     fIsOutputConstant = false;
                     break;
                  }
               }
            }
            if (fIsOutputConstant) {
               model.AddConstantTensor(fNY, fShapeY, outData);
               if (model.Verbose())
                  std::cout << ComparisionTrait<T, Op>::Name() << " op ---> " << fNY << "  "
                            << ConvertShapeToString(fShapeY) << " : " << ConvertValuesToString(length, outData)
                            << " (constant) " << std::endl;
            }
         }
         delete[] outData;
         // case of non constant output (no constant or shape tensors)
         if (!fIsOutputConstant && !fShapeY.empty()) {
            model.AddIntermediateTensor(fNY, ETensorType::BOOL, fShapeY);
            fDimShapeY = ConvertShapeToDim(fShapeY);
            if (model.Verbose())
               std::cout << ComparisionTrait<T, Op>::Name() << " op ---> " << fNY << "  "
                         << ConvertShapeToString(fShapeY) << std::endl;
         }
      } else {
         // case of dynamic tensors
          // case A or B have dynamic shapes. We need to broadcast if shape are not same
         auto ret = UTILITY::MultidirectionalBroadcastShape(fDimShapeX1, fDimShapeX2);
         fBroadcastFlag = ret.first;
         fDimShapeY = ret.second;
         // case of all parametric shapes and MultiDirectionalBroadcastShape  return the max of the 2
         // need to do before we declare the output tensor shape and the broadcasted ones
         if (ret.first & 4) {
            // check if one of the parameter is an input dimension
            // define function to find this
            auto IsInputDimParam = [&](const std::string &p) {
               auto inputNames = model.GetInputTensorNames();
               for (auto &input : inputNames) {
                  for (auto &i_s : model.GetDimTensorShape(input)) {
                     if (i_s.isParam && i_s.param == p)
                        return true;
                  }
               }
               return false;
            };
            for (size_t i = 0; i < fDimShapeY.size(); i++) {
               auto &s = fDimShapeY[i];
               if (s.isParam && s.param.find("std::max") != std::string::npos) {
                  if (IsInputDimParam(fDimShapeX1[i].param)) {
                     // case dim is 1 we indicate that the input parameter is equal to 1
                     if (fDimShapeX1[i].dim != 1)
                        s = fDimShapeX1[i];
                     else
                        s = fDimShapeX2[i];
                  } else if (IsInputDimParam(fDimShapeX2[i].param)) {
                     if (fDimShapeX2[i].dim != 1)
                        s = fDimShapeX2[i];
                     else
                        s = fDimShapeX1[i];
                  }
               }
            }
         }

         model.AddIntermediateTensor(fNY, ETensorType::BOOL, fDimShapeY);
         if (model.Verbose()) {
            std::cout << ComparisionTrait<T, Op>::Name()  << " : " << fNX1 << "  " << ConvertShapeToString(fDimShapeX1) << " , "
                                                          << fNX2 << "  " << ConvertShapeToString(fDimShapeX2) << " --> "
                                                          << fNY  << "  " << ConvertShapeToString(fDimShapeY) << std::endl;
            model.PrintIntermediateTensors();
         }
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

     if (fDimShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << ComparisionTrait<T,Op>::Name() << "  " << opName
                                 << " --> " << ConvertShapeToString(fShapeY) << "\n";

      // need to add check if tensors are compatible as in binary operator

      // use same code as Binary operator
      auto stridesA = UTILITY::ComputeStrideFromShape(fDimShapeX1);
      auto stridesB = UTILITY::ComputeStrideFromShape(fDimShapeX2);
      auto stridesY = UTILITY::ComputeStrideFromShape(fDimShapeY);

      std::string compute_idx_X1, compute_idx_X2, compute_idx_Y;
      if (fDimShapeX1.empty() ||
          std::all_of(fDimShapeX1.begin(), fDimShapeX1.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
         compute_idx_X1 = "0";
      } else {
         for (size_t i = 0; i < fDimShapeX1.size(); ++i) {
            if (fDimShapeX1[i].dim == 1 || fDimShapeX1[i].GetVal() == "1")
               continue;
            compute_idx_X1 += "idx_" + std::to_string(i + (fDimShapeY.size() - fDimShapeX1.size()));
            if (stridesA[i].GetVal() != "1")
               compute_idx_X1 += " * " + stridesA[i].GetVal();
            compute_idx_X1 += " + ";
         }
         // remove last 3 character " + "
         for (int j = 0; j < 3; j++)
            compute_idx_X1.pop_back();
      }
      if (fDimShapeX2.empty() ||
          std::all_of(fDimShapeX2.begin(), fDimShapeX2.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
         compute_idx_X2 = "0";
      } else {
         for (size_t i = 0; i < fDimShapeX2.size(); ++i) {
            if (fDimShapeX2[i].dim == 1 || fDimShapeX2[i].GetVal() == "1")
               continue;
            compute_idx_X2 += "idx_" + std::to_string(i + (fDimShapeY.size() - fDimShapeX2.size()));
            if (stridesB[i].GetVal() != "1")
               compute_idx_X2 += " * " + stridesB[i].GetVal();
            compute_idx_X2 += " + ";
         }
          // remove last 3 character " + "
         for (int j = 0; j < 3; j++)
            compute_idx_X2.pop_back();
      }
      int nloop = 0;
      if (fDimShapeY.empty() ||
          std::all_of(fDimShapeY.begin(), fDimShapeY.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
         compute_idx_Y = "0";
      } else {
         for (size_t i = 0; i < fDimShapeY.size(); ++i) {
            if (fDimShapeY[i].dim != 1 && fDimShapeY[i].GetVal() != "1") {
               nloop++;
               for (int j = 0; j < nloop; j++) out << SP;
               out << "for (size_t idx_" << i << " = 0; idx_" << i << " < " << fDimShapeY[i]
                   << "; ++idx_" << i << "){\n";
               compute_idx_Y += "idx_" + std::to_string(i);
               if (stridesY[i].GetVal() != "1")
                  compute_idx_Y += " * " + stridesY[i].GetVal();
               compute_idx_Y += " + ";
            }
         }
         // remove last 3 characters " + "
         for (int j = 0; j < 3; j++)
            compute_idx_Y.pop_back();
      }
      for (int j = 0; j < nloop + 1; j++) out << SP;
      out << "tensor_" << fNY << "[" << compute_idx_Y << "] = "
          << ComparisionTrait<T,Op>::Op( "tensor_" + fNX1 + "[" + compute_idx_X1 + "]" ,
                                         "tensor_" + fNX2 + "[" + compute_idx_X2 + "]") <<  " ;\n";


      for (int i = nloop; i > 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
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
