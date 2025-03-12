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
   std::vector<size_t> fShape;

   std::string fType;

public:
   ROperator_Softmax() {}
   ROperator_Softmax(int64_t attr_axis, std::string nameX, std::string nameY)
      : fAttrAxis(attr_axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input)
   {
      auto ret = input; // suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) ==
          false) { // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Softmax Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Softmax -> " << fNY << " " << ConvertShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName)
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Softmax called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShape.size();
      size_t length = ConvertShapeToLength(fShape);
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
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
         out << SP << SP << "tensor_" << fNY << "[i] /= sum;\n";
         out << SP << "}\n";
      } else {
         size_t batch = fShape[0];
         size_t channel = fShape[1];
         size_t width = (size > 2) ? fShape[size - 1] : 1;
         size_t height = (size > 3) ? fShape[size - 2] : 1;
         size_t depth = (size > 4) ? fShape[size - 3] : 1;
         size_t hStride = width;
         size_t dStride = height * width;
         size_t cStride = depth * dStride;
         size_t bStride = channel * cStride;

         size_t N = 0; // Size of the axis
         size_t iStride = 0;
         if (axis == 0) {
            N = batch;
            iStride = bStride;
         } else if (axis == 1) {
            N = channel;
            iStride = cStride;
         } else if (axis == size - 1) {
            N = width;
            iStride = 1;
         } else if (size > 3 && axis == size - 2) {
            N = height;
            iStride = hStride;
         } else if (size == 5 && axis == size - 3) {
            N = depth;
            iStride = dStride;
         } else {
            throw
               std::runtime_error("TMVA::SOFIE - Softmax operator along the axis "
                  + std::to_string(fAttrAxis) + " with " + std::to_string(size)
                  + "d input tensor not supported.");
         }

         bool notBatch = axis != 0;
         bool notChannel = axis != 1;
         bool notDepth = (size == 5 && axis != 2);
         bool notHeight = (size == 5 && axis != 3) || (size == 4 && axis != 2);
         bool notWidth = (size == 5 && axis != 4) || (size == 4 && axis != 3) || (size == 3 && axis != 2);

         if (notBatch) {
            out << SP << "for (size_t n = 0; n < " << batch << " ; n++){\n";
         }
         if (notChannel) {
            out << SP << SP << "for (size_t c = 0; c < " << channel << " ; c++){\n";
         }
         if (notDepth) {
            out << SP << SP << "for (size_t d = 0; d < " << depth << " ; d++){\n";
         }
         if (notHeight) {
            out << SP << SP << "for (size_t h = 0; h < " << height << " ; h++){\n";
         }
         if (notWidth) {
            out << SP << SP << "for (size_t w = 0; w < " << width << " ; w++){\n";
         }
         out << SP << SP << SP << fType << " sum = 0.;\n";
         out << SP << SP << SP << "size_t index = 0";
         if (notBatch) {
            out << " + n * " << bStride;
         }
         if (notChannel) {
            out << "+ c * " << cStride;
         }
         if (notDepth) {
            out << " + d * " << dStride;
         }
         if (notHeight) {
            out << " + h * " << hStride;
         }
         if (notWidth) {
            out << " + w";
         }
         out << ";\n";
         // apply softmax along the axis - find first maximum value for numerical stability
         if (N == 0)
            throw std::runtime_error("TMVA::SOFIE - Softmax operator is along axis with zero elements");
         out << SP << SP << SP << fType << " vmax = tensor_" << fNX << "[index];\n";
         out << SP << SP << SP << "for (size_t i = 1; i < " << N << "; i++) {\n";
         out << SP << SP << SP << SP << "if (tensor_" << fNX << "[index + i*" << iStride << "] > vmax)\n";
         out << SP << SP << SP << SP << SP << "vmax = tensor_" << fNX << "[index + i*" << iStride << "];\n";
         out << SP << SP << SP << "}\n";
         out << SP << SP << SP << "for (size_t i = 0; i < " << N << "; i++) {\n";
         out << SP << SP << SP << SP << "tensor_" << fNY << "[index + i*" << iStride << "] = std::exp(tensor_" << fNX
             << "[index + i*" << iStride << "] - vmax);\n";
         out << SP << SP << SP << SP << "sum += tensor_" << fNY << "[index + i*" << iStride << "];\n";
         out << SP << SP << SP << "}\n";
         out << SP << SP << SP << "for (size_t i = 0; i < " << N << "; i++) {\n";
         out << SP << SP << SP << SP << "tensor_" << fNY << "[index + i*" << iStride << "] /= sum;\n";
         out << SP << SP << SP << "}\n";
         if (notWidth) {
            out << SP << SP << "}\n"; // end w
         }
         if (notHeight) {
            out << SP << SP << "}\n"; // end h
         }
         if (notDepth) {
            out << SP << SP << "}\n"; // end d
         }
         if (notChannel) {
            out << SP << SP << "}\n"; // end c
         }
         if (notBatch) {
            out << SP << "}\n"; // end n
         }
      }
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_Softmax
