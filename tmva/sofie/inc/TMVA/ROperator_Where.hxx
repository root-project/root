#ifndef TMVA_SOFIE_ROperator_Where
#define TMVA_SOFIE_ROperator_Where

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Where final : public ROperator {
private:

   bool fIsInputBoolTensor = false;

   // Tensor names: C = condition, X = true branch, Y = false branch, Z = output
   std::string fNC;            // condition (bool)
   std::string fNX;            // true-branch values
   std::string fNY;            // false-branch values
   std::string fNZ;            // output
   std::string fNBroadcastedC;
   std::string fNBroadcastedX;
   std::string fNBroadcastedY;

   // Static shapes (used when all inputs are non-dynamic)
   std::vector<size_t> fShapeC;
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeZ;

   // Dynamic shapes (Dim-aware, used when any input is dynamic)
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
   ROperator_Where() {}
   ROperator_Where(const std::string &nameC,
                   const std::string &nameX,
                   const std::string &nameY,
                   const std::string &nameZ)
      : fNC(UTILITY::Clean_name(nameC)),
        fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY)),
        fNZ(UTILITY::Clean_name(nameZ))
   {
      fInputTensorNames  = { fNC, fNX, fNY };
      fOutputTensorNames = { fNZ };
   }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   {
      // output type follows X (and Y), not C (which is bool)
      return { input[1] };
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      // conservative: assume same shape (broadcasting resolved in Initialize)
      return { input[1] };
   }

   void Initialize(RModel &model) override
   {
      // ---------------------------------------------------------------- //
      //  Check all inputs exist
      // ---------------------------------------------------------------- //
      if (!model.CheckIfTensorAlreadyExist(fNC))
         throw std::runtime_error(std::string("TMVA SOFIE Where Op: condition tensor ") + fNC + " not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error(std::string("TMVA SOFIE Where Op: X tensor ") + fNX + " not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNY))
         throw std::runtime_error(std::string("TMVA SOFIE Where Op: Y tensor ") + fNY + " not found in model");

      // condition tensor is bool (uint8) - mark if it is a live input tensor
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
            std::cout << "Where : X " << fNX << " is dynamic " << ConvertDimShapeToString(fDimShapeX) << "\n";
         if (dynamicInputs & 4)
            std::cout << "Where : Y " << fNY << " is dynamic " << ConvertDimShapeToString(fDimShapeY) << "\n";
      }

      // ---------------------------------------------------------------- //
      //  Static path: all shapes known at code-gen time
      // ---------------------------------------------------------------- //
      if (dynamicInputs == 0) {

         // Multidirectional broadcast over all three tensors
         auto retXY = UTILITY::MultidirectionalBroadcastShape(fShapeX, fShapeY);
         fBroadcastFlag = retXY.first;
         fShapeZ = retXY.second;
         // also factor in C
         auto retCZ = UTILITY::MultidirectionalBroadcastShape(fShapeC, fShapeZ);
         fBroadcastFlag |= retCZ.first;
         fShapeZ = retCZ.second;

         bool allConstant = model.IsConstantTensor(fNC) &&
                            model.IsConstantTensor(fNX) &&
                            model.IsConstantTensor(fNY);

         if (allConstant) {
            // ----------------------------------------------------------
            //  Constant folding: evaluate Where at model initialisation
            // ----------------------------------------------------------
            auto broadcastIfNeeded = [&](const std::string &name,
                                         const std::vector<size_t> &shape,
                                         std::string &bcName,
                                         const std::string &prefix) {
               if (shape != fShapeZ) {
                  bcName = prefix + name + "to" + fNZ;
                  auto data = model.GetInitializedTensorData(name);
                  std::shared_ptr<void> bcData(
                     UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), shape, fShapeZ),
                     std::default_delete<T[]>());
                  model.AddConstantTensor(bcName, model.GetTensorType(name), fShapeZ, bcData);
               }
            };

            broadcastIfNeeded(fNX, fShapeX, fNBroadcastedX, "BC_");
            broadcastIfNeeded(fNY, fShapeY, fNBroadcastedY, "BC_");
            broadcastIfNeeded(fNC, fShapeC, fNBroadcastedC, "BC_");

            const std::string &nameC = fNBroadcastedC.empty() ? fNC : fNBroadcastedC;
            const std::string &nameX = fNBroadcastedX.empty() ? fNX : fNBroadcastedX;
            const std::string &nameY = fNBroadcastedY.empty() ? fNY : fNBroadcastedY;

            auto dataC = static_cast<bool *>(model.GetInitializedTensorData(nameC).get());
            auto dataX = static_cast<T *>   (model.GetInitializedTensorData(nameX).get());
            auto dataY = static_cast<T *>   (model.GetInitializedTensorData(nameY).get());

            size_t len = ConvertShapeToLength(fShapeZ);
            std::vector<T> dataZ(len);
            for (size_t i = 0; i < len; ++i)
               dataZ[i] = dataC[i] ? dataX[i] : dataY[i];

            model.AddConstantTensor<T>(fNZ, fShapeZ, dataZ.data());
            model.SetNotWritableInitializedTensor(nameC);
            model.SetNotWritableInitializedTensor(nameX);
            model.SetNotWritableInitializedTensor(nameY);
            fIsOutputConstant = true;
            fOutputTensorNames.pop_back();

            if (model.Verbose())
               std::cout << "Where --> " << fNZ << " " << ConvertShapeToString(fShapeZ)
                         << " : " << ConvertValuesToString(dataZ) << " (constant)\n";
         } else {
            // ----------------------------------------------------------
            //  Non-constant static: register broadcasted intermediates
            // ----------------------------------------------------------
            auto registerBC = [&](const std::string &name,
                                  const std::vector<size_t> &shape,
                                  std::string &bcName,
                                  const std::string &prefix) {
               if (shape != fShapeZ) {
                  bcName = prefix + name + "to" + fNZ;
                  if (model.IsInitializedTensor(name)) {
                     auto data = model.GetInitializedTensorData(name);
                     std::shared_ptr<void> bcData(
                        UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), shape, fShapeZ),
                        std::default_delete<T[]>());
                     model.AddConstantTensor(bcName, model.GetTensorType(name), fShapeZ, bcData);
                  } else {
                     model.AddIntermediateTensor(bcName, model.GetTensorType(name), fShapeZ);
                  }
               }
            };

            registerBC(fNX, fShapeX, fNBroadcastedX, "BC_");
            registerBC(fNY, fShapeY, fNBroadcastedY, "BC_");
            registerBC(fNC, fShapeC, fNBroadcastedC, "BC_");

            fDimShapeZ = ConvertShapeToDim(fShapeZ);
            model.AddIntermediateTensor(fNZ, model.GetTensorType(fNX), fShapeZ);

            if (model.Verbose())
               std::cout << "Where : C=" << fNC << " " << ConvertShapeToString(fShapeC)
                         << "  X=" << fNX << " " << ConvertShapeToString(fShapeX)
                         << "  Y=" << fNY << " " << ConvertShapeToString(fShapeY)
                         << " --> Z=" << fNZ << " " << ConvertShapeToString(fShapeZ) << "\n";
         }

      } else {
         // ---------------------------------------------------------------- //
         //  Dynamic path: at least one input has a parametric shape
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
                  // prefer X dim over Y dim
                  if (i < fDimShapeX.size() && IsInputDimParam(fDimShapeX[i].param)) {
                     s = (fDimShapeX[i].dim != 1) ? fDimShapeX[i] : fDimShapeY[i];
                  } else if (i < fDimShapeY.size() && IsInputDimParam(fDimShapeY[i].param)) {
                     s = (fDimShapeY[i].dim != 1) ? fDimShapeY[i] : fDimShapeX[i];
                  }
               }
            }
         }

         model.AddIntermediateTensor(fNZ, model.GetTensorType(fNX), fDimShapeZ);

         if (model.Verbose())
            std::cout << "Where (dynamic) : C=" << ConvertDimShapeToString(fDimShapeC)
                      << "  X=" << ConvertDimShapeToString(fDimShapeX)
                      << "  Y=" << ConvertDimShapeToString(fDimShapeY)
                      << " --> Z=" << ConvertDimShapeToString(fDimShapeZ) << "\n";
      }
   }

   std::string GenerateInitCode() override
   {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override
   {
      if (fIsOutputConstant) return "";

      opName = "op_" + opName;

      if (fDimShapeZ.empty()) {
         throw std::runtime_error("TMVA SOFIE Where Op called to Generate without being initialized first");
      }

      std::stringstream out;
      out << SP << "\n//------ WHERE " << opName << " --> " << ConvertDimShapeToString(fDimShapeZ) << "\n";

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
                   << "throw std::runtime_error(\"SOFIE Where: cannot broadcast X dim " << i << " in " << opName << "\");\n";
            }
            // validate Y vs Z
            if (i < fDimShapeY.size() && fDimShapeY[i].isParam) {
               out << SP << SP << "if (" << fDimShapeY[i] << " != 1 && "
                   << fDimShapeY[i] << " != " << fDimShapeZ[i] << ")\n";
               out << SP << SP << SP
                   << "throw std::runtime_error(\"SOFIE Where: cannot broadcast Y dim " << i << " in " << opName << "\");\n";
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

      // ---------------------------------------------------------------- //
      //  Runtime broadcasting for non-constant, non-initialised tensors
      // ---------------------------------------------------------------- //
      // Broadcast X if needed
      if (!fNBroadcastedX.empty()) {
         out << SP << "// Broadcast X tensor " << fNX << "\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<"
             << TensorType<T>::Name() << ">(tensor_" << fNX << ", "
             << ConvertDimShapeToString(fDimShapeX) << ", "
             << ConvertDimShapeToString(fDimShapeZ) << ", tensor_" << fNBroadcastedX << ");\n";
      }
      // Broadcast Y if needed
      if (!fNBroadcastedY.empty()) {
         out << SP << "// Broadcast Y tensor " << fNY << "\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<"
             << TensorType<T>::Name() << ">(tensor_" << fNY << ", "
             << ConvertDimShapeToString(fDimShapeY) << ", "
             << ConvertDimShapeToString(fDimShapeZ) << ", tensor_" << fNBroadcastedY << ");\n";
      }
      // Broadcast C (condition) if needed
      if (!fNBroadcastedC.empty()) {
         if (fIsInputBoolTensor) {
            // live bool input: need a temporary std::vector for the broadcast utility
            size_t inputLength = ConvertShapeToLength(fShapeC);
            out << SP << "std::vector<std::uint8_t> tmp_tensor_" << fNC
                << "(tensor_" << fNC << ", tensor_" << fNC << " + " << inputLength << ");\n";
            out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<std::uint8_t>"
                << "(tmp_tensor_" << fNC << ".data(), "
                << ConvertDimShapeToString(fDimShapeC) << ", "
                << ConvertDimShapeToString(fDimShapeZ) << ", tensor_" << fNBroadcastedC << ");\n";
         } else {
            out << SP << "// Broadcast condition tensor " << fNC << "\n";
            out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<std::uint8_t>"
                << "(tensor_" << fNC << ", "
                << ConvertDimShapeToString(fDimShapeC) << ", "
                << ConvertDimShapeToString(fDimShapeZ) << ", tensor_" << fNBroadcastedC << ");\n";
         }
      }

      // Final (possibly broadcasted) tensor names
      const std::string nameX = fNBroadcastedX.empty() ? fNX : fNBroadcastedX;
      const std::string nameY = fNBroadcastedY.empty() ? fNY : fNBroadcastedY;
      const std::string nameC = fNBroadcastedC.empty() ? fNC : fNBroadcastedC;

      // ---------------------------------------------------------------- //
      //  Generate loop(s) with per-dimension stride-based index arithmetic
      //  (same pattern as BasicBinary)
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
          << "tensor_" << nameC << "[" << idxC << "] ? "
          << "tensor_" << nameX << "[" << idxX << "] : "
          << "tensor_" << nameY << "[" << idxY << "];\n";

      // Close loops
      for (int i = nloop; i > 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
      }

      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROperator_Where
