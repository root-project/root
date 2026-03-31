#ifndef TMVA_SOFIE_ROPERATOR_CLIP
#define TMVA_SOFIE_ROPERATOR_CLIP

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// ---------------------------------------------------------------------------
// ROperator_Clip
//
// ONNX spec: Y = max(min_val, min(max_val, X))  element-wise
//
// The min and max bounds are optional in the ONNX spec:
//   - if fNMin is empty  → no lower clipping  (effectively -inf)
//   - if fNMax is empty  → no upper clipping  (effectively +inf)
//
// Bounds can be provided either as:
//   (a) initializer / constant tensors (scalar, shape []),
//   (b) runtime input tensors          (resolved at Generate time),
//   (c) compile-time float literals    (via the fMin / fMax attributes).
//
// The implementation follows the Selu operator style exactly:
//   - static shape stored in fShape
//   - dynamic shape stored in fDimShape
//   - a flat loop over all elements in Generate()
// ---------------------------------------------------------------------------

template <typename T>
class ROperator_Clip final : public ROperator {
private:

   // Tensor names
   std::string fNX;       // input data
   std::string fNY;       // output
   std::string fNMin;     // optional: tensor name for min bound
   std::string fNMax;     // optional: tensor name for max bound


   // Static shape (non-dynamic path, mirrors Selu)
   std::vector<size_t> fShape;

   // Dynamic shape (Dim-aware, for dynamic input tensors)
   std::vector<Dim> fDimShape;
   bool fIsDynamic = false;

   // Compile-time bound values — used when bounds are constant tensors
   // Initialised to the ONNX defaults (no clipping)
   T fMin =  std::numeric_limits<T>::lowest();   // -inf equivalent
   T fMax =  std::numeric_limits<T>::max();      //  +inf equivalent

   // Flags indicating whether each bound is:
   //   - absent (no input provided)
   //   - a constant resolved at Initialize time
   //   - a runtime tensor that must be read in the generated code
   bool fHasMin         = false;
   bool fHasMax         = false;
   bool fMinIsConstant  = false;
   bool fMaxIsConstant  = false;

public:

   ROperator_Clip() {}

   // Constructor for the common case where bounds are tensor inputs
   // (follows ONNX node input order: X, min, max)
   ROperator_Clip(std::string nameX,
                  std::string nameY,
                  std::string nameMin = "",
                  std::string nameMax = "")
      : fNX  (UTILITY::Clean_name(nameX)),
        fNY  (UTILITY::Clean_name(nameY)),
        fNMin(nameMin.empty() ? "" : UTILITY::Clean_name(nameMin)),
        fNMax(nameMax.empty() ? "" : UTILITY::Clean_name(nameMax))
   {
      fInputTensorNames  = { fNX };
      if (!fNMin.empty()) fInputTensorNames.push_back(fNMin);
      if (!fNMax.empty()) fInputTensorNames.push_back(fNMax);
      fOutputTensorNames = { fNY };
   }

   // Convenience constructor when bounds are known scalars at model-build time
   ROperator_Clip(std::string nameX,
                  std::string nameY,
                  T minVal,
                  T maxVal)
      : fNX (UTILITY::Clean_name(nameX)),
        fNY (UTILITY::Clean_name(nameY)),
        fMin(minVal), fMax(maxVal),
        fHasMin(true), fHasMax(true),
        fMinIsConstant(true), fMaxIsConstant(true)
   {
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }


   // -----------------------------------------------------------------------
   void Initialize(RModel& model) override
   {
      // ---- validate main input ------------------------------------------
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error(
            "TMVA SOFIE Clip Op Input Tensor " + fNX + " is not found in model");

      // ---- collect shape (static or dynamic, mirrors BasicBinary) -------
      if (model.IsDynamicTensor(fNX)) {
         fIsDynamic = true;
         fDimShape  = model.GetDynamicTensorShape(fNX);
      } else {
         fShape    = model.GetTensorShape(fNX);
         fDimShape = ConvertShapeToDim(fShape);
      }

      // ---- resolve min bound --------------------------------------------
      if (!fNMin.empty() && model.CheckIfTensorAlreadyExist(fNMin)) {
         fHasMin = true;
         if (model.IsInitializedTensor(fNMin)) {
            // constant scalar tensor — read value now
            auto data = static_cast<T*>(model.GetInitializedTensorData(fNMin).get());
            fMin            = data[0];
            fMinIsConstant  = true;
            model.SetNotWritableInitializedTensor(fNMin);
         }
         // else: runtime input — will be dereferenced in generated code
      }

      // ---- resolve max bound --------------------------------------------
      if (!fNMax.empty() && model.CheckIfTensorAlreadyExist(fNMax)) {
         fHasMax = true;
         if (model.IsInitializedTensor(fNMax)) {
            auto data = static_cast<T*>(model.GetInitializedTensorData(fNMax).get());
            fMax            = data[0];
            fMaxIsConstant  = true;
            model.SetNotWritableInitializedTensor(fNMax);
         }
      }

      // ---- register output tensor ---------------------------------------
      if (fIsDynamic)
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fDimShape);
      else
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "Clip : " << fNX << " "
                   << ConvertShapeToString(fShape);
         if (fHasMin)
            std::cout << "  min=" << (fMinIsConstant
                       ? std::to_string(fMin) : fNMin + "(runtime)");
         if (fHasMax)
            std::cout << "  max=" << (fMaxIsConstant
                       ? std::to_string(fMax) : fNMax + "(runtime)");
         std::cout << " --> " << fNY << "\n";
      }

      // only needs <algorithm> and <limits> — no cmath
      model.AddNeededStdLib("algorithm");
      model.AddNeededStdLib("limits");
   }


   // -----------------------------------------------------------------------
   // Generate
   // -----------------------------------------------------------------------
   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;

      if (fShape.empty() && fDimShape.empty())
         throw std::runtime_error(
            "TMVA SOFIE Operator Clip called to Generate without being initialized first");

      std::stringstream out;
      out << SP << "\n//------ CLIP " << OpName << "\n";

      // ---- build the length expression (static or dynamic) -------------
      std::string length = ConvertDimShapeToLength(fDimShape);

      // ---- build min/max expressions for the generated code ------------
      //
      //  Priority:
      //    1. compile-time constant value  → emit literal
      //    2. runtime input tensor         → emit tensor_<name>[0]  (scalar)
      //    3. not provided                 → emit numeric_limits extreme
      //
      std::string minExpr, maxExpr;

      if (fMinIsConstant) {
         minExpr = ToStringHighPrec(fMin);
      } else if (fHasMin) {
         minExpr = "tensor_" + fNMin + "[0]";  // scalar input tensor
      } else {
         // No lower bound — use lowest representable value
         minExpr = "std::numeric_limits<" + TensorType<T>::Name()
                   + ">::lowest()";
      }

      if (fMaxIsConstant) {
         maxExpr = ToStringHighPrec(fMax);
      } else if (fHasMax) {
         maxExpr = "tensor_" + fNMax + "[0]";
      } else {
         // No upper bound — use max representable value
         maxExpr = "std::numeric_limits<" + TensorType<T>::Name()
                   + ">::max()";
      }

      auto tensorValue = [](const std::string & name, const std::string & index) {
         std::stringstream s;
         s << "tensor_" << name << "[" << index << "]";
         return s.str();
      };

      // ---- flat element loop (identical structure to Selu) -------------
      out << SP << "for (int id = 0; id < " << length << " ; id++) {\n";
      std::string firstExpr = fHasMax ? "std::min(" + maxExpr + ", " + tensorValue(fNX, "id") + ")" : tensorValue(fNX, "id");
      std::string secondExpr  = fHasMin ? "std::max(" + minExpr + ", " + firstExpr + ")" : firstExpr;
      out << SP << SP << tensorValue(fNY, "id") << " = " << secondExpr << ";\n";
      out << SP << "}\n";

      return out.str();
   }


private:

   // Helper: convert a T value to string with enough precision
   std::string ToStringHighPrec(T val) const {
      std::ostringstream ss;
      ss << std::setprecision(std::numeric_limits<T>::max_digits10) << val;
      // add dot if missing
      if (ss.str().find(".") == std::string::npos) ss << ".";
      // append 'f' suffix for float literals so generated code compiles
      // cleanly without implicit double→float conversion warnings
      if (std::is_same<T, float>::value) ss << "f";
      return ss.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_CLIP
