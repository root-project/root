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


   std::string fNX;
   std::string fNY;
   std::string fNC;
   std::string fNBroadcastedX;
   std::string fNBroadcastedY;
   std::string fNBroadcastedC;
   std::string fNZ;



   // static shapes (used when tensors are not dynamic) )
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeC;
   std::vector<size_t> fShapeZ;

   // Dynamic generic shapes
   std::vector<Dim> fDimShapeC;
   std::vector<Dim> fDimShapeX;
   std::vector<Dim> fDimShapeY;
   std::vector<Dim> fDimShapeZ;

   // Broadcast flag: mirrors convention of BasicBinary
   //   bit 0: broadcast Y->X (Y needs expanding)
   //   bit 1: broadcast X->Y (X needs expanding)
   //   bit 2: broadcast C->Z (C needs expanding)
   //   bit 4: shapes may differ at runtime (dynamic)
   int fBroadcastFlag = 0;

public:
   ROperator_Where(){}
   ROperator_Where(const std::string & nameC, const std::string & nameX, const std::string & nameY, const std::string & nameZ):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fNC(UTILITY::Clean_name(nameC)), fNZ(UTILITY::Clean_name(nameZ)){
         fInputTensorNames = { fNX, fNY, fNC };
         fOutputTensorNames = { fNZ };
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
      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNX + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNY)) {
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNY + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNC)) {
         throw std::runtime_error(std::string("TMVA SOFIE Where Op Input Tensor ") + fNC + "is not found in model");
      }
      // check if fNC input tensor is boolean
      if (model.IsReadyInputTensor(fNC))
         fIsInputBoolTensor = true;

      // ---------------------------------------------------------------- //
      //  Collect shapes – dynamic or static
      // ---------------------------------------------------------------- //
      int dynamicInputs = 0;   // bitmask: bit0=C, bit1=X, bit2=Y

      if (model.IsDynamicTensor(fNC)) {
         fDimShapeC = model.GetDynamicTensorShape(fNC);
         dynamicInputs |= 1;
      } else {
         fShapeC    = model.GetTensorShape(fNC);
         fDimShapeC = ConvertShapeToDim(fShapeC);
      }
      if (model.IsDynamicTensor(fNX)) {
         fDimShapeX = model.GetDynamicTensorShape(fNX);
         dynamicInputs |= 2;
      } else {
         fShapeX    = model.GetTensorShape(fNX);
         fDimShapeX = ConvertShapeToDim(fShapeX);
      }
      if (model.IsDynamicTensor(fNY)) {
         fDimShapeY = model.GetDynamicTensorShape(fNY);
         dynamicInputs |= 4;
      } else {
         fShapeY    = model.GetTensorShape(fNY);
         fDimShapeY = ConvertShapeToDim(fShapeY);
      }


      if (model.Verbose()) {
         if (dynamicInputs & 1)
            std::cout << "Where : condition " << fNC << " is dynamic " << ConvertDimShapeToString(fDimShapeC) << "\n";
         if (dynamicInputs & 2)
            std::cout << "Where :  " << fNX << " is dynamic " << ConvertDimShapeToString(fDimShapeX) << "\n";
         if (dynamicInputs & 4)
            std::cout << "Where : Y " << fNZ << " is dynamic " << ConvertDimShapeToString(fDimShapeZ) << "\n";
      }

      // ---------------------------------------------------------------- //
      //  Static path: all shapes known at code-gen time
      // ---------------------------------------------------------------- //
      if (dynamicInputs == 0) {

         bool broadcast = !UTILITY::AreSameShape(fShapeX, fShapeY) || !UTILITY::AreSameShape(fShapeX, fShapeC);
         if (broadcast) {
            // find shape to broadcast between X,Y,C looking for max length
            size_t lengthX = ConvertShapeToLength(fShapeX);
            size_t lengthY = ConvertShapeToLength(fShapeY);
            size_t lengthC = ConvertShapeToLength(fShapeC);
            bool broadcastX = false, broadcastY = false, broadcastC = false;
            if (lengthX >= lengthY && lengthX >= lengthC) {
               fShapeZ = fShapeX;
               // broadcast Y and C if different than X
               broadcastY = (lengthY != lengthX);
               broadcastC = (lengthC != lengthX);
            } else if (lengthY >= lengthX && lengthY >= lengthC) {
               fShapeZ = fShapeY;
               // broadcast X and C if different than Y
               broadcastX = (lengthX != lengthY);
               broadcastC = (lengthC != lengthY);
            } else if (lengthC >= lengthX && lengthC >= lengthY) {
               fShapeZ = fShapeC;
               // broadcast X and Y if different than C
               broadcastX = (lengthX != lengthC);
               broadcastY = (lengthY != lengthC);
            }

            // Broadcast X to Z
            if (broadcastX) {
               fNBroadcastedX = "BC_" + fNX + "_to_" + fNZ;
               if (model.IsInitializedTensor(fNX)) {
                  auto data = model.GetInitializedTensorData(fNX);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), fShapeX, fShapeZ),
                     std::default_delete<T[]>());
                  // Update the data and the shape of X
                  model.AddConstantTensor(fNBroadcastedX, model.GetTensorType(fNX), fShapeZ, broadcastedData);
                  fShapeX = fShapeZ;
               } else {
                  // I need to prepend to shape of X the extra dimensions added for broadcasting to Z
                  if (fShapeX.size() < fShapeZ.size()) {
                     size_t nPrepend = fShapeZ.size() - fShapeX.size();
                     fShapeX.insert(fShapeX.begin(), nPrepend, 1);
                  }
               }
            }
            // Broadcast Y to Z
            if (broadcastY) {
               fNBroadcastedY = "BC_" + fNY + "_to_" + fNZ;
               if (model.IsInitializedTensor(fNY)) {
                  auto data = model.GetInitializedTensorData(fNY);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), fShapeY, fShapeZ),
                     std::default_delete<T[]>());
                  // do not update tensor B but add broadcasted one (since it can be input to some other operators)
                  model.AddConstantTensor(fNBroadcastedY, model.GetTensorType(fNY), fShapeZ, broadcastedData);
                  fShapeY = fShapeZ;
               } else {
                  // I need to prepend to shape of Y the extra dimensions added for broadcasting to Z
                  if (fShapeY.size() < fShapeZ.size()) {
                     size_t nPrepend = fShapeZ.size() - fShapeY.size();
                     fShapeY.insert(fShapeY.begin(), nPrepend, 1);
                  }

               }
            }
            // Broadcast C to Z
            if (broadcastC) {
               fNBroadcastedC = "BC_" + fNC + "_to_" + fNZ;
               if (model.IsInitializedTensor(fNC)) {
                  auto data = model.GetInitializedTensorData(fNC);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), fShapeC, fShapeZ),
                     std::default_delete<T[]>());
                  // do not update tensor C but add broadcasted one (since it can be input to some other operators)
                  model.AddConstantTensor(fNBroadcastedC, model.GetTensorType(fNC), fShapeZ, broadcastedData);
                  fShapeC = fShapeZ;
               } else {
                  // I need to prepend to shape of C the extra dimensions added for broadcasting to Z
                  if (fShapeC.size() < fShapeZ.size()) {
                     size_t nPrepend = fShapeZ.size() - fShapeC.size();
                     fShapeC.insert(fShapeC.begin(), nPrepend, 1);
                  }
               }
            }
         } else {
            fShapeZ = fShapeX;
         }
         // check case of constant  output (if all inputs are defined)
         if (model.IsInitializedTensor(fNC)) {
            std::string nameC = fNBroadcastedC.empty() ? fNC : fNBroadcastedC;
            auto dataC = static_cast<bool *>(model.GetInitializedTensorData(nameC).get());
            model.SetNotWritableInitializedTensor(nameC);
            T *dataX = nullptr;
            T *dataY = nullptr;
            std::vector<Dim> shapeDataX;
            std::vector<Dim> shapeDataY;
            if (model.IsInitializedTensor(fNX)) {
               std::string nameX = fNBroadcastedX.empty() ? fNX : fNBroadcastedX;
               dataX = static_cast<T *>(model.GetInitializedTensorData(nameX).get());
               // flag tensors to not be written in a file
               model.SetNotWritableInitializedTensor(nameX);
            } else if (model.IsShapeTensor(fNX)) {
               shapeDataX = model.GetShapeTensorValues(fNX);
            }
            if (model.IsInitializedTensor(fNY)) {
               std::string nameY = fNBroadcastedY.empty() ? fNY : fNBroadcastedY;
               dataY = static_cast<T *>(model.GetInitializedTensorData(nameY).get());
               model.SetNotWritableInitializedTensor(nameY);
            } else if (model.IsShapeTensor(fNY)) {
               shapeDataY = model.GetShapeTensorValues(fNY);
            }
            std::vector<T> dataZ;        // used in case output is constant tensor
            std::vector<Dim> shapeDataZ; // used in case output is a shape tensor (can be also constant if all
                                         // dimensions are not parametric)
            // if fNC (condition) is initialized we know the output is a shape or a constant tensor,
            // so we can compute it at initialization and add it as a constant tensor to the model
            // (and not add the operator output as intermediate tensor to the model)
            bool isOutputConstantTensor = true;
            if (dataX && dataY) {
               dataZ.resize(ConvertShapeToLength(fShapeZ));
               for (size_t i = 0; i < dataZ.size(); i++)
                  dataZ[i] = (dataC[i]) ? dataX[i] : dataY[i];
               if (model.Verbose())
                  std::cout << "data A and B : dataZ constant: " << ConvertValuesToString(dataZ) << std::endl;
            } else if (dataX && shapeDataY.size() > 0) {
               shapeDataZ.resize(ConvertShapeToLength(fShapeZ));
               for (size_t i = 0; i < shapeDataZ.size(); i++) {
                  shapeDataZ[i] = (dataC[i]) ? Dim{size_t(dataX[i])} : shapeDataY[i];
                  isOutputConstantTensor &= !shapeDataZ[i].isParam;
               }
               if (model.Verbose())
                  std::cout << "data A but shapeB " << ConvertDimShapeToString(shapeDataY) << "  "
                         << isOutputConstantTensor << std::endl;
            } else if (dataY && shapeDataX.size() > 0) {
               shapeDataZ.resize(ConvertShapeToLength(fShapeZ));
               for (size_t i = 0; i < shapeDataZ.size(); i++) {
                  shapeDataZ[i] = (dataC[i]) ? shapeDataY[i] : Dim{size_t(dataY[i])};
                  isOutputConstantTensor &= !shapeDataZ[i].isParam;
               }
               if (model.Verbose())
                  std::cout << "data B but shapeA " << ConvertDimShapeToString(shapeDataX) << "  "
                         << isOutputConstantTensor << std::endl;
            } else if (shapeDataY.size() > 0 && shapeDataX.size() > 0) {
               shapeDataZ.resize(ConvertShapeToLength(fShapeZ));
               for (size_t i = 0; i < shapeDataZ.size(); i++) {
                  shapeDataZ[i] = (dataC[i]) ? shapeDataX[i] : shapeDataY[i];
                  isOutputConstantTensor &= !shapeDataZ[i].isParam;
               }
               if (model.Verbose())
                  std::cout << " shapeA and B " << ConvertDimShapeToString(shapeDataX) << " shapeB "
                         << ConvertDimShapeToString(shapeDataY) << "  " << isOutputConstantTensor << std::endl;
            }
            fIsOutputConstant = true;
            // add as constant or shape tensor depending on the case
            if (dataZ.size() > 0)
               model.AddConstantTensor<T>(fNZ, fShapeZ, dataZ.data());
            else if (shapeDataZ.size() > 0)
               model.AddShapeTensor(fNZ, shapeDataZ, fShapeZ.size() == 0);
            else {
               fIsOutputConstant = false;
            }
            if (fIsOutputConstant && model.Verbose())
               std::cout << "Where op ---> " << fNZ << "  " << ConvertShapeToString(fShapeZ) << " : "
                         << ((dataZ.size() > 0) ? ConvertValuesToString(dataZ) : ConvertDimShapeToString(shapeDataZ))
                         << ((dataZ.size() > 0) ? " (constant)" : " (shape)") << std::endl;

            // output is a constant tensor
            if (fIsOutputConstant)
               fOutputTensorNames.pop_back();
         }
         if (!fIsOutputConstant) {

            fDimShapeZ = ConvertShapeToDim(fShapeZ);
            model.AddIntermediateTensor(fNZ, model.GetTensorType(fNX), fShapeZ);
            if (model.Verbose())
               std::cout << "Where : condition : " << fNC << "  " << ConvertShapeToString(fShapeC) << " X "
                         << fNX << "  " << ConvertShapeToString(fShapeX) << " Y " << fNY << "  "
                         << ConvertShapeToString(fShapeY) << " ---> " << fNZ << "  " << ConvertShapeToString(fShapeZ)
                         << std::endl;
         }
      } else {
         // ---------------------------------------------------------------- //
         //  Dynamic path: at least one input has a parametric shape
         //  Need to use BroadcastShape to find output shape
         // ---------------------------------------------------------------- //
         auto retXY = UTILITY::MultidirectionalBroadcastShape(fDimShapeX, fDimShapeY);
         fBroadcastFlag = retXY.first;
         fDimShapeZ     = retXY.second;
         auto retCZ = UTILITY::MultidirectionalBroadcastShape(fDimShapeC, fDimShapeZ);
         fBroadcastFlag |= retCZ.first;
         fDimShapeZ      = retCZ.second;

         // Resolve std::max params to actual input dim params (same logic as BasicBinary)
         if (fBroadcastFlag & 4) {
            auto IsInputDimParam = [&](const std::string &p) {
               for (auto &input : model.GetInputTensorNames())
                  for (auto &s : model.GetDimTensorShape(input))
                     if (s.isParam && s.param == p) return true;
               return false;
            };
            for (size_t i = 0; i < fDimShapeZ.size(); i++) {
               auto &s = fDimShapeZ[i];
               if (s.isParam && s.param.find("std::max") != std::string::npos) {
                  // prefer A dim over B dim
                  if (i < fDimShapeX.size() && IsInputDimParam(fDimShapeX[i].param)) {
                     s = (fDimShapeX[i].dim != 1) ? fDimShapeX[i] : fDimShapeY[i];
                  } else if (i < fDimShapeY.size() && IsInputDimParam(fDimShapeY[i].param)) {
                     s = (fDimShapeY[i].dim != 1) ? fDimShapeY[i] : fDimShapeX[i];
                  }
               }
            }
         }
         // I need to prepend to shape of X,Y,C the extra dimensions added for broadcasting to Z
         if (fDimShapeX.size() < fDimShapeZ.size()) {
            size_t nPrepend = fDimShapeZ.size() - fDimShapeX.size();
            fDimShapeX.insert(fDimShapeX.begin(), nPrepend, Dim{1});
         }
         if (fDimShapeY.size() < fDimShapeZ.size()) {
            size_t nPrepend = fDimShapeZ.size() - fDimShapeY.size();
            fDimShapeY.insert(fDimShapeY.begin(), nPrepend, Dim{1});
         }
         if (fDimShapeC.size() < fDimShapeZ.size()) {
            size_t nPrepend = fDimShapeZ.size() - fDimShapeC.size();
            fDimShapeC.insert(fDimShapeC.begin(), nPrepend, Dim{1});
         }

         model.AddIntermediateTensor(fNZ, model.GetTensorType(fNX), fDimShapeZ);

         if (model.Verbose())
            std::cout << "Where (dynamic) : C=" << ConvertDimShapeToString(fDimShapeC)
                      << "  A=" << ConvertDimShapeToString(fDimShapeX)
                      << "  B=" << ConvertDimShapeToString(fDimShapeY)
                      << " --> Y=" << ConvertDimShapeToString(fDimShapeZ) << "\n";
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override {

      opName = "op_" + opName;
      std::stringstream out;
      out << SP << "\n//------ WHERE " << opName << " --> " << ConvertDimShapeToString(fDimShapeZ) << "\n";
      if (fIsOutputConstant) return out.str();


      // ---------------------------------------------------------------- //
      //  Runtime broadcast validation (dynamic shapes, flag bit 4)
      // ---------------------------------------------------------------- //
      if (fBroadcastFlag & 4) {
         auto lengthX = ConvertDimShapeToLength(fDimShapeX);
         auto lengthY = ConvertDimShapeToLength(fDimShapeY);
         auto lengthC = ConvertDimShapeToLength(fDimShapeC);
         out << SP << "if (" << lengthX << " != " << lengthY << " || "
             << lengthX << " != " << lengthC << ") {\n";
         for (size_t i = 0; i < fDimShapeZ.size(); i++) {
            // validate X vs Z
            if (i < fDimShapeX.size() && fDimShapeX[i].isParam) {
               out << SP << SP << "if (" << fDimShapeX[i] << " != 1 && "
                   << fDimShapeX[i] << " != " << fDimShapeZ[i] << ")\n";
               out << SP << SP << SP
                   << "throw std::runtime_error(\"SOFIE Where: cannot broadcast A dim " << i << " in " << opName << "\");\n";
            }
            // validate Y vs Z
            if (i < fDimShapeY.size() && fDimShapeY[i].isParam) {
               out << SP << SP << "if (" << fDimShapeY[i] << " != 1 && "
                   << fDimShapeY[i] << " != " << fDimShapeZ[i] << ")\n";
               out << SP << SP << SP
                   << "throw std::runtime_error(\"SOFIE Where: cannot broadcast B dim " << i << " in " << opName << "\");\n";
            }
            // validate C vs Z
            if (i < fDimShapeC.size() && fDimShapeC[i].isParam) {
               out << SP << SP << "if (" << fDimShapeC[i] << " != 1 && "
                   << fDimShapeC[i] << " != " << fDimShapeZ[i] << ")\n";
               out << SP << SP << SP
                   << "throw std::runtime_error(\"SOFIE Where: cannot broadcast C dim " << i << " in " << opName << "\");\n";
            }
         }
         out << SP << "}\n";
      }
      // implement now where using teh strides and looping on the different dimensions
      // ---------------------------------------------------------------- //
      //  Generate loop(s) with per-dimension stride-based index arithmetic
      // ---------------------------------------------------------------- //
      auto stridesX = UTILITY::ComputeStrideFromShape(fDimShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fDimShapeY);
      auto stridesC = UTILITY::ComputeStrideFromShape(fDimShapeC);
      auto stridesZ = UTILITY::ComputeStrideFromShape(fDimShapeZ);

      auto buildIdxExpr = [&](const std::vector<Dim> &dimShape,
                               const std::vector<Dim> &strides,
                               size_t rankZ) -> std::string {
         if (dimShape.empty() ||
             std::all_of(dimShape.begin(), dimShape.end(),
                         [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; }))
            return "0";
         std::string expr;
         size_t offset = rankZ - dimShape.size();
         for (size_t i = 0; i < dimShape.size(); ++i) {
            if (dimShape[i].dim == 1 || dimShape[i].GetVal() == "1") continue;
            expr += "idx_" + std::to_string(i + offset);
            if (strides[i].GetVal() != "1")
               expr += " * " + strides[i].GetVal();
            expr += " + ";
         }
         if (expr.size() >= 3)
            for (int j = 0; j < 3; j++) expr.pop_back();  // remove trailing " + "
         return expr.empty() ? "0" : expr;
      };

      std::string idxX = buildIdxExpr(fDimShapeX, stridesX, fDimShapeZ.size());
      std::string idxY = buildIdxExpr(fDimShapeY, stridesY, fDimShapeZ.size());
      std::string idxC = buildIdxExpr(fDimShapeC, stridesC, fDimShapeZ.size());

       // Emit nested loops over output shape
      int nloop = 0;
      std::string idxZ;
      // case Z is a scalar (all dimensions are 1) or Z has no dimension
      if (fDimShapeZ.empty() ||
          std::all_of(fDimShapeZ.begin(), fDimShapeZ.end(),
                      [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
         idxZ = "0";
      } else {
         for (size_t i = 0; i < fDimShapeZ.size(); ++i) {
            if (fDimShapeZ[i].dim != 1 && fDimShapeZ[i].GetVal() != "1") {
               nloop++;
               for (int j = 0; j < nloop; j++) out << SP;
               out << "for (size_t idx_" << i << " = 0; idx_" << i
                   << " < " << fDimShapeZ[i] << "; ++idx_" << i << ") {\n";
               idxZ += "idx_" + std::to_string(i);
               if (stridesZ[i].GetVal() != "1")
                  idxZ += " * " + stridesZ[i].GetVal();
               idxZ += " + ";
            }
         }
         if (idxZ.size() >= 3)
            for (int j = 0; j < 3; j++) idxZ.pop_back();
      }

      // Inner assignment
      for (int j = 0; j < nloop + 1; j++) out << SP;
      out << "tensor_" << fNZ << "[" << idxZ << "] = "
          << "tensor_" << fNC << "[" << idxC << "] ? "
          << "tensor_" << fNX << "[" << idxX << "] : "
          << "tensor_" << fNY << "[" << idxY << "];\n";

      // Close loops
      for (int i = nloop; i > 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
      }

      return out.str();
   }


};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Where
