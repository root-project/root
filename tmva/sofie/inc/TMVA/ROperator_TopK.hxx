#ifndef TMVA_SOFIE_ROPERATOR_TOPK
#define TMVA_SOFIE_ROPERATOR_TOPK

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   Dim fK;
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   std::string fType;

public:
   ROperator_TopK() {}
   ROperator_TopK(int attr_axis, int attr_largest, int attr_sorted, std::string nameK, std::string nameX, std::string nameVal, std::string nameInd)
      : fAttrAxis(attr_axis),
        fAttrLargest(attr_largest),
        fAttrSorted(attr_sorted),
        fNK(UTILITY::Clean_name(nameK)),
        fNX(UTILITY::Clean_name(nameX)),
        fNVal(UTILITY::Clean_name(nameVal)),
        fNInd(UTILITY::Clean_name(nameInd)){
            fInputTensorNames = { fNX, fNK };
            fOutputTensorNames = { fNVal, fNInd };
        }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType ret = input[0];
      return {ret, ret};
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);
      auto fShapeK = model.GetTensorShape(fNK);
      auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      size_t kval = *kptr;
      model.SetNotWritableInitializedTensor(fNK);
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if(static_cast<size_t>(fAttrAxis) >=  fShapeX.size()){
         throw
            std::runtime_error("TMVA::SOFIE ONNX TopK op axis = "+ std::to_string(fAttrAxis) +" value exeeds size of tensor " +fNX+" of size "+fShapeX.size()+" .");
      }
      // fK cannot be larger that axis dimension
      if (fShapeX[fAttrAxis].isParam)
         fK = Dim{std::string("std::min(size_t(" + std::to_string(kval) + "), " + fShapeX[fAttrAxis].GetVal() + ")" ), static_cast<size_t>(-1) };
      else
         fK = Dim { std::min(kval, fShapeX[fAttrAxis].dim) };

      // output shape is equal to input shape apart for value in fAttrAxis
      fShapeY = fShapeX;
      fShapeY[fAttrAxis] = Dim{fK};

      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);

      // output indices should be an int64 tensor
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));

      if (model.Verbose()) {
         std::cout << "TopK " << fNX << "  " << ConvertShapeToString(fShapeX)
                      << "---> " << fNVal << " " <<  ConvertShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShapeX.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ TopK\n";

      auto length=ConvertDimShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      // we perform loop on dimension before sorted axis and after sorted axis
      std::vector<Dim> shape_before(fShapeX.begin(), fShapeX.begin() + axis);   // input shape before axis
      std::string n_before = (axis>0) ? ConvertDimShapeToLength(shape_before) : "1";
      std::string n_after = strideX[axis].GetVal();
      std::string n_elements = fShapeX[axis].GetVal(); // number of elements to be sorted

      // }
      out << SP << "{\n"; // to define a separate scope for the operator code
      out << SP << "std::vector<std::pair<float,int64_t>> elements(" << n_elements << ");\n";
      // loop on elements before
      if (n_before != "1") {
         out << SP << "for (size_t i = 0; i < " << n_before << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << strideX[axis-1] << ";\n";
         out << SP << SP << "size_t yoffset = i*" << strideY[axis-1] << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (n_after !=  "1")
         out << SP << "for (size_t j = 0; j < " << n_after << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      // copy elements to be sorted in vector of pair
      out << SP << SP << "for (size_t l = 0; l < " << n_elements << "; l++) {\n";
      out << SP << SP << SP << "elements[l] = std::make_pair(tensor_" << fNX << "[xoffset + " << strideX[axis] << "*l + j], l);\n";
      out << SP << SP << "}\n";

      if (fAttrSorted) {
         if (fAttrLargest) {
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
               "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});\n";

         } else
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
            "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first<b.first) : a.second < b.second;});\n";
      } else
         // in this case we don;t need to return sorted elements, so we keep same order as before
         out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end());\n";

      // copy the selected elements in the output
      out << SP << SP << "for (size_t l = 0; l < " << fK << "; l++) {\n";
      out << SP << SP << SP << "tensor_" << fNVal   << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].first;\n";
      out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].second;\n";
      out << SP << SP << "}\n";
      if (n_after != "1") out << SP << SP << "}\n";
      if (n_before != "1") out << SP << "}\n";
      out << SP << "}\n"; // end operator scope
      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_ROPERATOR_TOPK
