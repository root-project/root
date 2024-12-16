#ifndef TMVA_SOFIE_ROperator_Einsum
#define TMVA_SOFIE_ROperator_Einsum

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

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

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);

      if (model.Verbose()) {
         std::cout << "Einstein op ";
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

   std::string Generate(std::string OpName) override {

      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;

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

      // loops on the output indices  i0,....iN
      int outDims = fShapeY.size();
      int inDims = fSumLabels.length();
      assert(outDims == fOutputLabels.size());
      assert(inDims == fSumDims.size());
      for (int i = 0; i < outDims; i++) {
         for (int j = 0; j < i; j++) out << SP;
         std::string l {fOutputLabels[i]};
         out << "for (int " << l << " = 0; " << l << " < " << fShapeY[i] << "; " << l << "++) {\n";
      }
      // reset to zero output tensor
      auto outputStride = UTILITY::ComputeStrideFromShape(fShapeY);
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

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Einsum
