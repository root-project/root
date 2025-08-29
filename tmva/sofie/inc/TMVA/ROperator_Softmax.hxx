#ifndef TMVA_SOFIE_ROPERATOR_Softmax
#define TMVA_SOFIE_ROPERATOR_Softmax

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_Softmax final : public ROperator {

private:
   int64_t fAttrAxis;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

   std::string fType;

public:
   ROperator_Softmax() {}
   ROperator_Softmax(int64_t attr_axis, std::string nameX, std::string nameY)
      : fAttrAxis(attr_axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; // suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) ==
          false) { // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Softmax Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Softmax -> " << fNY << " " << ConvertShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Softmax called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShape.size();
      auto length_str = ConvertDimShapeToLength(fShape);
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;

      // Check if this is the special case where memory is contiguous.
      if (axis == size - 1) {
         std::string axis_size = fShape[axis].GetVal();
         std::string num_rows;
         if (IsInteger(length_str) && IsInteger(axis_size)) {
            num_rows = std::to_string(std::stoul(length_str) / std::stoul(axis_size));
         } else {
            num_rows = "(" + length_str + ") / (" + axis_size + ")";
         }
         
         out << "\n" << SP << "//------ SOFTMAX - " << size << "  " << length_str << "  " << axis << "\n";
         out << SP << "for (int i = 0; i < " << num_rows << "; ++i) {\n";
         out << SP << SP << "size_t offset = i * " << axis_size << ";\n";
         out << SP << SP << fType << " const * x_ptr = &tensor_" << fNX << "[offset];\n";
         out << SP << SP << fType << " * y_ptr = &tensor_" << fNY << "[offset];\n";
         
         out << SP << SP << fType << " vmax = x_ptr[0];\n";
         out << SP << SP << "for (int j = 1; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "if (x_ptr[j] > vmax) vmax = x_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " sum = 0.0;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] = std::exp(x_ptr[j] - vmax);\n";
         out << SP << SP << SP << "sum += y_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " inv_sum = 1.0f / sum;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] *= inv_sum;\n";
         out << SP << SP << "}\n";
         out << SP << "}\n";

      } else {
         auto stride = UTILITY::ComputeStrideFromShape(fShape);
         size_t k = 0;
         std::vector<std::string> l(size);
         for (size_t i = 0; i < size; i++) {
            if (i != axis) {
               for (size_t j = 0; j < k; j++) out << SP;
               l[i] = std::string("i") + std::to_string(i);
               out << "for (int " << l[i] << " = 0; " << l[i] << " < " << fShape[i] << "; " << l[i] << "++) {\n";
               k++;
            }
         }
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " sum = 0.;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "size_t index = ";
         bool first = true;
         for (size_t i = 0; i < size; i++) {
            if (i == axis) continue;
            if (!first) out << " + ";
            if (stride[i].GetVal() != "1")
               out << stride[i] << "*";
            out << l[i];
            first = false;
         }
         out << ";\n";
         // find maximum looping along reduced axis
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " vmax = tensor_" << fNX << "[index];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 1; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << fType << " x = tensor_" << fNX << "[index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << "];\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "if (x > vmax) vmax = x;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // compute softmax
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] = std::exp(tensor_" << fNX << "[id] - vmax);\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "sum += tensor_" << fNY << "[id];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // normalize
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
          for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << "] /= sum;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         //end loops
         for (int i = static_cast<int>(k) - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }
   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_Softmax
