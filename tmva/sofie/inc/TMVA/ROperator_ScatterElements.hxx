#ifndef TMVA_SOFIE_ROperator_ScatterElements
#define TMVA_SOFIE_ROperator_ScatterElements

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{


class ROperator_ScatterElements final : public ROperator{
private:

   int64_t fAxis;

   std::string fNX;
   std::string fNI;
   std::string fNU;
   std::string fNY;
   std::string fReduction;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeI;
   std::vector<size_t> fShapeY;

   // define reduction function. Possibilities are:
   // none (default), add, mul, max, min
   std::string ReductionFunction(const std::string & t1, const std::string & t2 ) {
      std::string name = fReduction;
      if (name.empty() || name == "none")
         return t2;
      else if (name == "add")
         return t1 + " + " + t2;
      else if (name == "mul")
         return t1 + " * " + t2;
      else if (name == "max")
         return "std::max(" + t1 + "," + t2 + ")";
      else if (name == "min")
         return "std::min(" + t1 + "," + t2 + ")";
      else
         throw std::runtime_error("TMVA SOFIE ScatterElements : invalid reduction attribute");

      return std::string();
   }

public:
   ROperator_ScatterElements(){}
   ROperator_ScatterElements(const std::string & nameX, const std::string & nameI, const std::string & nameU, const std::string & nameY,
                           int axis, std::string reduction):
      fAxis(axis),
      fNX(UTILITY::Clean_name(nameX)), fNI(UTILITY::Clean_name(nameI)), fNU(UTILITY::Clean_name(nameU)),
      fNY(UTILITY::Clean_name(nameY)),
      fReduction(reduction)
      {
         fInputTensorNames = { fNX, fNI, fNU };
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements Op Input Tensor ") + fNX + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNI)) {
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements Op Input Tensor ") + fNI + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNU)) {
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements Op Input Tensor ") + fNU + "is not found in model");
      }
      //tbd check for constant tensors

      fShapeX = model.GetTensorShape(fNX);
      fShapeI = model.GetTensorShape(fNI);
      if (model.GetTensorShape(fNU) != fShapeI)
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements - update tensor has invalid shape ")) ;
      if (fShapeX.size() == 0)
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements - input tensor has zero rank  ")) ;
      if (fShapeX.size() != fShapeI.size())
         throw std::runtime_error(std::string("TMVA SOFIE ScatterElements - index tensor has invalid rank  ")) ;

      if (fAxis < 0) fAxis += fShapeX.size();

      // assume output shape is identical to input shape
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override {

      if (fIsOutputConstant) return "";

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE ScatterElements Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//-------- ScatterElements  --- " << opName << "\n";

      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto strideI = UTILITY::ComputeStrideFromShape(fShapeI);

      size_t length = ConvertShapeToLength(fShapeY);

      // function to write compute expression for global index from axes indices
      auto tensorIndex = [](const std::vector<size_t> & stride, const std::vector<std::string> & idx) {
         std::stringstream strst;
         int dims = idx.size();
         assert (dims == (int) stride.size());
         for (int i = 0; i < dims; i++) {
            if (stride[i] != 1)
               strst << stride[i] << "*" << idx[i];
            else
               strst << idx[i];
            if (i < dims-1)
               strst << " + ";
         }
         return strst.str();
      };


      // copy first input in output (maybe can be avoided??)
      out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";

      // loop on tensor rank
      int dims = fShapeY.size();
      std::vector<std::string> idx(dims);
      for (int i = 0; i < dims; i++) {
         idx[i] = std::string("i") + std::to_string(i);
         for (int j = 0; j <= i; j++) out << SP;
         out << "for (int " << idx[i] << " = 0; " << idx[i] << " < " << fShapeI[i] << "; " << idx[i] << "++) {\n";
      }
      // correct index for specific axis
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int updateIndex = " << tensorIndex(strideI,idx) << ";\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int iAxis = tensor_" << fNI << "[updateIndex];\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "if (iAxis < 0) iAxis += " << fShapeY[fAxis] << ";\n";
      idx[fAxis] = "iAxis";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int  outIndex = " << tensorIndex(strideY, idx) << ";\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "tensor_" << fNY << "[outIndex] = "
         << ReductionFunction(std::string("tensor_") + fNY + "[outIndex]", std::string("tensor_") + fNU + "[updateIndex]") << ";\n";

      for (int i = dims; i > 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
      }
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_ScatterElements
