#ifndef TMVA_SOFIE_ROPERATOR_Softmax
#define TMVA_SOFIE_ROPERATOR_Softmax

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// implement Softmax and LogSoftmax
class ROperator_Softmax final : public ROperator {

private:
   bool fLogSoftmax;  // for the logsoftmax case
   int64_t fAttrAxis;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

   std::string fType;

public:
   ROperator_Softmax() {}
   ROperator_Softmax(int64_t attr_axis, std::string nameX, std::string nameY, bool logSoftmax)
      : fLogSoftmax(logSoftmax),
      fAttrAxis(attr_axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))

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
      int size = fShape.size();
      auto length = ConvertDimShapeToLength(fShape);
      auto stride = UTILITY::ComputeStrideFromShape(fShape);
      int axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ SOFTMAX - " << size << "  " << length << "  " << axis << "\n";
      // use safe numerically implementation by subtracting max of tensor
      if (size == 1) {
         out << SP << fType << " vmax = tensor_" << fNX << "[0];\n";
         out << SP << "for (size_t i = 1; i < " << length << " ; i++){\n";
         out << SP << SP << "if (tensor_" << fNX << "[i] > vmax) vmax = tensor_" << fNX << "[i];\n";
         out << SP << "}\n";
         out << SP << fType << " sum = 0.0;\n";
         out << SP << "for (size_t i = 0; i < " << length << " ; i++){\n";
         out << SP << SP << "tensor_" << fNY << "[i] = std::exp(tensor_" << fNX << "[i] - vmax);\n";
         out << SP << SP << "sum += tensor_" << fNY << "[i];\n";
         out << SP << "}\n";
         out << SP << "for (size_t i = 0; i < " << length << " ; i++){\n";
         if (fLogSoftmax)
            out << SP << SP << "tensor_" << fNY << "[i] = std::log(tensor_" << fNY << "[i] / sum );\n";
         else
            out << SP << SP << "tensor_" << fNY << "[i] /= sum;\n";
         out << SP << "}\n";
      } else {
         int k = 0;
         std::vector<std::string> l(size);
         for (int i = 0; i < size; i++) {
            if (i != axis) {
               for (int j = 0; j < k; j++) out << SP;
               l[i] = std::string("i") + std::to_string(i);
               out << "for (int " << l[i] << " = 0; " << l[i] << " < " << fShape[i] << "; " << l[i] << "++) {\n";
               k++;
            }
         }
         for (int j = 0; j < size-1; j++) out << SP;
         out << fType << " sum = 0.;\n";
         for (int j = 0; j < size-1; j++) out << SP;
         out << "size_t index = ";
         for (int i = 0; i < size; i++) {
            if (i == axis) continue;
            if ((i > 0 && axis != 0) || i > 1 ) out << "+";
            if (stride[i].GetVal() != "1")
               out << stride[i] << "*";
            out << l[i];
         }
         out << ";\n";
         // find maximum looping along reduced axix
         for (int j = 0; j < size-1; j++) out << SP;
         out << fType << " vmax = tensor_" << fNX << "[index];\n";
         for (int j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (int j = 0; j < size; j++) out << SP;
         out << fType << " x = tensor_" << fNX << "[index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << "];\n";
         for (int j = 0; j < size; j++) out << SP;
         out << "if (x > vmax) vmax = x;\n";
         for (int j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // compute softmax
         for (int j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (int j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (int j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] = std::exp(tensor_" << fNX << "[id] - vmax);\n";
         for (int j = 0; j < size; j++) out << SP;
         out << "sum += tensor_" << fNY << "[id];\n";
         for (int j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // normalize
         for (int j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (int j = 0; j < size; j++) out << SP;
         // define the tensor y value for given i index
         std::string tensor_y_i = "tensor_" + fNY + "[index + i";
         if (stride[axis].GetVal() != "1")
            tensor_y_i += "*(" + stride[axis].GetVal() + ")";
         tensor_y_i += "]";
         if (fLogSoftmax) {
            out << tensor_y_i << " = std::log(" << tensor_y_i << " / sum);\n";
         } else {
            out << tensor_y_i << " /= sum;\n";
         }
         for (int j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         //end loops
         for (int i = size-2; i >=0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_Softmax
