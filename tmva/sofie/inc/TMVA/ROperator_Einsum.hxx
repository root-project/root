#ifndef TMVA_SOFIE_ROperator_Einsum
#define TMVA_SOFIE_ROperator_Einsum

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{



template<typename T>
class ROperator_Einsum final : public ROperator{
private:

   bool fIsInputBoolTensor = false;


   std::vector<std::string> fNInputs;
   std::string fNY;

   std::vector<std::string> fInputLabels;
   std::string fOutputLabels;
   std::string fSumLabels;  // string containing the reducing labels
   std::string fGemmType;

   std::vector<int> fSumDims; // dimension of the labels we use to perform summing

   std::vector<std::vector<size_t>> fShapeInputs;
   std::vector<size_t> fShapeY;




public:
   ROperator_Einsum(){}
   ROperator_Einsum(const std::string & equation, const std::vector<std::string> & namesX, const std::string & nameY):
      fNInputs(namesX.size()), fNY(UTILITY::Clean_name(nameY))
   {
      for (size_t i = 0; i < namesX.size(); i++)
         fNInputs[i] = UTILITY::Clean_name(namesX[i]);

      // parse teh equations to find labels
      if (!ParseEquation(equation))
         throw std::runtime_error("TMVA SOFIE Einsum Op: Error parsing the equation " + equation);

      fInputTensorNames.resize(fNInputs.size());
      std::transform(fNInputs.begin(), fNInputs.end(), fInputTensorNames.begin(),
                  [](const std::string& s) -> std::string { return s; });
      fOutputTensorNames = { fNY };
   }

   bool ParseEquation(const std::string & input_equation) {
      std::string eq (input_equation);
      // remove blank spaces
      eq.erase(std::remove(eq.begin(), eq.end(), ' '), eq.end());
      // look for '->'  finding the first occurrence
      std::string target("->");
      size_t pos = eq.find(target);
      if (pos == std::string::npos) {
         std::cout << "'->' not found in the equation." << std::endl;
         return false;
      }
      // Substring before the target
      std::string inputStr = eq.substr(0, pos);
      // Substring after the target
      std::string outputStr = eq.substr(pos + target.length());

      // look now for the group of labels separated by "," in the inputs
      size_t start = 0;
      size_t pos1 = 0;
      // Extract labels separated by commas
      while ((pos1 = inputStr.find(',', start)) != std::string::npos) {
         std::string labels = inputStr.substr(start, pos1 - start);
         fInputLabels.push_back(labels);
         start = pos1 + 1; // Move past the comma
      }
      // Add the last label (after the final comma)
      fInputLabels.push_back(inputStr.substr(start));

      // check if labels are ok and do not contain alphanumeric characters
      auto checkLabel = [](const std::string & label) {
         for (char c : label) {
            if (!std::isalnum(c)) {
               std::cout << "Wrong tensor label " << label << std::endl;
               return false;
            }
         }
         // empty label is OK , is a scalar
         return true;
      };
      for (auto & label : fInputLabels) {
         if (!checkLabel(label)) return false;
      }
      if (!checkLabel(outputStr)) {
         std::cout << "invalid output label" << std::endl;
         return false;
      }
      fOutputLabels = outputStr;

      if (fInputLabels.size() != fNInputs.size()) {
         std::cout << "Invalid number of input labels found " << fInputLabels.size() << " for #inputs = " << fNInputs.size() << std::endl;
         return false;
      }
      // ignore for the time being broadcasting, empty output label and other features
      return true;
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
      size_t i = 0;
      std::map<char, int> labelsMap;
      for ( auto & name : fNInputs) {
         if (!model.CheckIfTensorAlreadyExist(name))
            throw std::runtime_error(std::string("TMVA SOFIE Einsum Op Input Tensor ") + name + "is not found in model");

         // if (model.IsDynamicTensor(name) || model.IsDimInputTensor(name) ) {
         //    // not yet supported
         // } else {
         auto shape = model.GetTensorShape(name);
         fShapeInputs.push_back(shape);
         //}
         // fill the label maps
         std::string labels = fInputLabels[i];
         for (size_t j = 0; j < shape.size(); j++) {
            if (j >= labels.length()) {
               throw std::runtime_error(std::string("TMVA SOFIE Einsum Op Input Tensor has invalid label or shape ") + labels + " " + ConvertShapeToString(shape));
            }
            labelsMap[labels[j]] = shape[j];
         }
         i++;
      }
      // get output shape from label maps
      for (char l : fOutputLabels) {
         if (labelsMap.count(l) == 0)
            throw std::runtime_error(std::string("TMVA SOFIE Einsum Op : output label ") + std::string(&l) + " is not present in inputs");
         fShapeY.push_back(labelsMap[l]);
      }
      // we need to get the labels we are going to sum
      // these are the labels not present in the output
      fSumLabels = "";
      fSumDims.clear();
      for (auto & l : labelsMap) {
         if (fOutputLabels.find(l.first) == std::string::npos) {
            fSumLabels += l.first;
            fSumDims.push_back(l.second);
         }
      }

      // check if we can use MatMul for EinSum
      // need to have one sum labels in the last 2 and have the first in common
      if (fNInputs.size() == 2 && fSumDims.size() == 1 && fShapeInputs[0].size() >=2 && fShapeInputs[1].size() >= 2 ) {
         // find positions of dum labels
         char l = fSumLabels[0];
         size_t pos1 = fInputLabels[0].find(l);
         size_t pos2 = fInputLabels[1].find(l);
         // check if summing is done in the last 2 indices of tensor

         if (pos1 == fInputLabels[0].length() - 1 && pos2 == fInputLabels[1].length() - 2)
            fGemmType = "nn";
         else if (pos1 == fInputLabels[0].length() - 2 && pos2 == fInputLabels[1].length() - 2)
            fGemmType = "tn";
         else if (pos1 == fInputLabels[0].length() - 1 && pos2 == fInputLabels[1].length() - 1)
            fGemmType = "nt";
         else if (pos1 == fInputLabels[0].length() - 2 && pos2 == fInputLabels[1].length() - 1)
            fGemmType = "tt";
         else
            fGemmType = "";
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);

      if (model.Verbose()) {
         std::cout << "Einsum op ";
         for (i = 0; i < fNInputs.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << fNInputs[i] << " " << ConvertShapeToString(fShapeInputs[i]) << " " << fInputLabels[i];
         }
         std::cout << " --> " << fNY << "  " << ConvertShapeToString(fShapeY) << "  " << fOutputLabels << std::endl;
      }

   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override {

      if (fIsOutputConstant) return "";

      opName = "op_" + opName;

      if (fShapeY.size() != fOutputLabels.length()) {
         throw std::runtime_error("TMVA SOFIE Einsum Op called to Generate without being initialized first");
      }

      // function to write compute expression index from strides
      auto tensorIndex = [](const std::vector<size_t> & stride, const std::string & labels) {
         std::stringstream strst;
         int dims = labels.length();
         // scalar case
         if (dims == 0) return std::string("0");
         assert (dims == (int) stride.size());
         for (int i = 0; i < dims-1; i++) {
            strst << stride[i] << "*" << std::string{labels[i]} << " + ";
         }
         strst << std::string{labels[dims-1]};
         return strst.str();
      };

      std::stringstream out;
      out << SP << "\n//-------- Einsum   \n";

      auto outputStride = UTILITY::ComputeStrideFromShape(fShapeY);

      // loops on the output indices  i0,....iN
      if (fGemmType.empty()) {
      int outDims = fShapeY.size();
      int inDims = fSumLabels.length();
      assert(outDims == int(fOutputLabels.size()));
      assert(inDims == int(fSumDims.size()));
      for (int i = 0; i < outDims; i++) {
         for (int j = 0; j < i; j++) out << SP;
         std::string l {fOutputLabels[i]};
         out << "for (int " << l << " = 0; " << l << " < " << fShapeY[i] << "; " << l << "++) {\n";
      }
      // reset to zero output tensor
      std::string outputIndex = tensorIndex(outputStride,fOutputLabels);

      for (int j = 0; j < outDims; j++) out << SP;
      out << "tensor_" << fNY << "[" << outputIndex << "] = 0;\n";
      // loop on remaining indices where we perform the sum
      for (int i = 0; i < inDims; i++) {
         for (int j = 0; j < outDims + i; j++) out << SP;
         std::string l {fSumLabels[i]};
         out << "for (int " << l << " = 0; " << l << " < " << fSumDims[i] << "; " << l << "++) {\n";
      }
      for (int j = 0; j < outDims+inDims; j++) out << SP;
      // tensor_out[outId] += t_in_0[ind0] * t_in1[ind1] *....
      out << "tensor_" << fNY << "[" << outputIndex << "] +=\n";
      for (size_t k = 0; k < fNInputs.size(); k++) {
         auto inputStride = UTILITY::ComputeStrideFromShape(fShapeInputs[k]);
         std::string inputIndex = tensorIndex(inputStride,fInputLabels[k]);
         for (int j = 0; j < outDims+inDims; j++) out << SP;
         out << SP << "tensor_" << fNInputs[k] << "[" << inputIndex << "]";
         if (fNInputs.size() > 1 && k < fNInputs.size() -1) out << " *\n";
      }
      out << ";\n";

      // end loops on all indices i0,....iN
      for (int i = outDims+inDims-1; i >= 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
      }


      } else {
         // case we use Gemm
         out << SP << "// implementing Einsum using MatMul   \n";
         // note A is second input and B first one - due to transpose of Fortran rep.
         out << SP << "char " << opName << "_transA = '" << fGemmType[0] << "';\n";
         out << SP << "char " << opName << "_transB = '" << fGemmType[1] << "';\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeInputs[0].size();
         int64_t dimB = fShapeInputs[1].size();

         auto m = (fGemmType[0] == 't') ? fShapeInputs[0][dimA-1] : fShapeInputs[0][dimA-2];
         auto n = (fGemmType[1] == 't') ? fShapeInputs[1][dimB-2] : fShapeInputs[1][dimB-1];
         auto k = (fGemmType[0] == 't') ? fShapeInputs[0][dimA-2] : fShapeInputs[0][dimA-1];

         out << SP << "int " << opName << "_m = " << m << ";\n";
         out << SP << "int " << opName << "_n = " << n << ";\n";
         out << SP << "int " << opName << "_k = " << k << ";\n";
         out << SP << "float " << opName << "_alpha = 1.0;\n";
         out << SP << "float " << opName << "_beta = 0.0;\n";
         out << SP << "int " << opName << "_lda = " << ((fGemmType[0] == 't') ? m : k) << ";\n";
         out << SP << "int " << opName << "_ldb = " << ((fGemmType[1] == 't') ? k : n) << ";\n";

         auto inputStrideA = UTILITY::ComputeStrideFromShape(fShapeInputs[0]);
         auto inputStrideB = UTILITY::ComputeStrideFromShape(fShapeInputs[1]);

         int stackDims = fShapeY.size()-2;
         for (int i = 0; i < stackDims; i++) {
            for (int j = 0; j < i; j++) out << SP;
            std::string l {fOutputLabels[i]};
            out << "for (int " << l << " = 0; " << l << " < " << fShapeY[i] << "; " << l << "++) {\n";
         }
         auto tensorOffset = [](const std::vector<size_t> & stride, const std::string & labels) {
            std::stringstream strst;
            int dims = labels.length()-2;
            // scalar case
            if (dims == 0) return std::string("0");
            assert (dims +2 == (int) stride.size());
            for (int i = 0; i < dims; i++) {
               strst << stride[i] << "*" << std::string{labels[i]};
               if (i < dims-1) strst << " + ";
            }
            return strst.str();
         };
         // only float type supported
         out << SP << "BLAS::sgemm_(&" << opName << "_transB, &" << opName << "_transA, &" << opName
             << "_n, &" << opName << "_m, &" << opName << "_k, &" << opName << "_alpha, "
             << "&tensor_" << fNInputs[1] << "[" << tensorOffset(inputStrideB, fInputLabels[1])
             << "], &" << opName << "_ldb, "
             << "&tensor_" << fNInputs[0] << "[" << tensorOffset(inputStrideA, fInputLabels[0]  ) << "], &" << opName << "_lda, &" << opName << "_beta, "
             << "&tensor_" << fNY << "[" << tensorOffset(outputStride,fOutputLabels) << "],  &" << opName << "_n);\n";


         for (int i = stackDims-1; i >= 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }

      }


      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override {
      return { std::string("Gemm") };
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Einsum
