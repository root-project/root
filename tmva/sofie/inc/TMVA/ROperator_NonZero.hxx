#ifndef TMVA_SOFIE_ROPERATOR_NONZERO
#define TMVA_SOFIE_ROPERATOR_NONZERO

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template<class T>
class ROperator_NonZero final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_NonZero(){}
   ROperator_NonZero(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }



   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE NonZero Op Input Tensor " + fNX + " is not found in model");
      }


      // case input is constant
      if (model.IsConstantTensor(fNX)) {
         // compute output directly
         T * data = static_cast<T*>(model.GetInitializedTensorData(fNX).get());
         // shape is fully known
         auto shapeX = model.GetTensorShape(fNX);
         std::vector<size_t> shapeY(2);
         shapeY[0] = shapeX.size();
         auto length = ConvertShapeToLength(shapeX);
         auto strides = UTILITY::ComputeStrideFromShape(shapeX);
         std::vector<std::vector<int64_t>> nonzero_indices;
         for (size_t i = 0; i < length; i++) {
            if (data[i] != 0) {
               // get indices
               size_t flat_index = i;
               std::vector<int64_t> indices(shapeX.size());
               for (size_t j = 0; j < shapeX.size(); ++j) {
                  indices[j] = flat_index / strides[j];
                  flat_index %= strides[j];
               }
               nonzero_indices.emplace_back(indices);
            }
         }
         shapeY[1] = nonzero_indices.size();
         std::vector<int64_t> dataY(shapeY[0]* shapeY[1]);
         size_t k = 0;
         for (size_t i = 0; i < shapeY[0]; i++) {
            for (size_t j = 0; j < shapeY[1]; j++) {
               dataY[k] = nonzero_indices[j][i];
               k++;
            }
         }
         if (dataY.empty()) {
            // no zero elements found
            dataY.resize(1);
            shapeY.clear();  // use an empty shape
         }

         model.AddConstantTensor(fNY, shapeY, dataY);
         if (model.Verbose()) {
            std::cout << "NonZero : " << fNX << " -> " << fNY << " " << ConvertShapeToString(shapeY)
                     << " : " << ConvertValuesToString(dataY) << std::endl;
         }
         fIsOutputConstant = true;

      } else {

         fShapeX = model.GetDimTensorShape(fNX);

         // output shape(-1) depends on number of elements of non zero values
         // first dim is rank of input
         fShapeY.resize(2);
         fShapeY[0] = fShapeX.size();

         // identify as -1 since we will declare maximum as size of input
         fShapeY[1] = Dim{std::string("v_NonZero_") + fNX, static_cast<size_t>(-1)};

         model.AddIntermediateTensor(fNY, ETensorType::INT64, fShapeY);
         if (model.Verbose()) {
            std::cout << "NonZero : " << fNX << " -> " << fNY << " " << ConvertShapeToString(fShapeY) << std::endl;
         }
      }
   }
   std::string GenerateSessionMembersCode(std::string /*opName*/) override {
      if (fIsOutputConstant) return "";
      // define output value used as max non zero with max size = input shape * N
      auto inputLength = ConvertDimShapeToLength(fShapeX);
      std::stringstream out;
      out << SP << "size_t v_NonZero_" << fNX << " = " << inputLength << ";\n";
      return out.str();
   }


   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         return "";
      }
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator NonZero called to Generate without being initialized first");
      }
      std::stringstream out;
      auto intShapeX = ConvertShapeToInt(fShapeX);
      size_t inputLength = 0;
      std::string s_inputLength = ConvertDimShapeToLength(fShapeX);
      if (!intShapeX.empty())
         inputLength = ConvertShapeToLength(intShapeX);

      size_t dims = fShapeX.size();
      out << "\n//------ NonZero\n";

      std::string vnonzero = "v_NonZero_" + fNX;

      // loop on input indices
      out << SP << "size_t offset_" << opName << " = 0;\n";
      out << SP << vnonzero << " = 0;\n";
      for (size_t j = 0; j < dims; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeX[j] << "; " << index << "++) {\n";
      }
      for (size_t k = 0; k <= dims; k++) out << SP;
      out << "if (tensor_" << fNX << "[offset_" << opName << "++]) {\n";
      for (size_t j = 0; j < dims; j++) {
         for (size_t k = 0; k <= dims+1; k++) out << SP;
         out << "tensor_" << fNY << "[";
         if (j > 0) {
            if (inputLength > 0) {
               out << inputLength * j;
            } else {
               out << s_inputLength;
               if (j > 1) out << " * " << j;
            }
            out << " + ";
         }
         out << vnonzero << "] = i_" << j << ";\n";
      }
      for (size_t k = 0; k <= dims+1; k++) out << SP;
      out << vnonzero << "++;\n";
      for (size_t k = 0; k <= dims; k++) out << SP;
      out << "}\n";
      //end loops
      for (size_t j = dims; j > 0; j--) {
         for (size_t k = 0; k <j; k++) out << SP;
         out << "}\n";
      }
      // now we need to rearrange the vector if nonzero is less than length of input
      out << SP << "if (" << vnonzero << " < " << s_inputLength << "){\n";
      for (size_t j = 1; j < dims; j++) {
         out << SP << SP << "std::copy(tensor_" << fNY;
         if (j>0) out << " + " << s_inputLength;
         if (j>1) out << " * " << j;
         out << ", tensor_" << fNY;
         if (j>0) out << " + " << s_inputLength;
         if (j>1) out << " * " << j;
         out << " + " << vnonzero << ", tensor_" <<  fNY;
         if (j>0) out << " + " << vnonzero;
         if (j>1) out << "* " << j;
         out << ");\n";
      }
      out << SP << "}\n";

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_NonZero
