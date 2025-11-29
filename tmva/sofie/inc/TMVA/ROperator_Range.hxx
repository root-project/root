#ifndef TMVA_SOFIE_ROPERATOR_RANGE
#define TMVA_SOFIE_ROPERATOR_RANGE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <algorithm>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Range final : public ROperator
{
private:

   std::string fNStart;
   std::string fNLimit;
   std::string fNDelta;
   std::string fNOutput;
   std::vector<Dim> fShape;
   std::string fType;

public:
   ROperator_Range(){}

   ROperator_Range(std::string start, std::string limit, std::string delta, std::string nameOutput):
      fNStart(start), fNLimit(limit), fNDelta(delta),
      fNOutput(UTILITY::Clean_name(nameOutput)) {
      if (std::is_same<T, float>::value) {
          fType = "float";
      } else if (std::is_same<T, int64_t>::value) {
          fType = "int64_t";
      }
      static_assert( (std::is_same_v<T, float> || std::is_same_v<T, int64_t>),
                  "TMVA::SOFIE - Unsupported type by Range operator");
      {
         fInputTensorNames = { fNStart, fNLimit, fNDelta };
         fOutputTensorNames = { fNOutput };
      }
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNStart)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNStart + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNLimit)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNLimit + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNDelta)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNDelta + "is not found in model");
      }
      ETensorType type = ConvertStringToType(fType);



      auto analyzeInput = [&](const std::string & tName, T & value, Dim & dim) {
         int ftype = 0; // type of input (0 intermediate, 1 constant , 2 shape)
         if (model.IsInitializedTensor(tName)) {
            T * data = static_cast<T*>(model.GetInitializedTensorData(tName).get());
            if (!data)
               std::runtime_error("TMVA SOFIE Range Op Input Tensor has invalid input  data");
            value = *data;
            ftype = 1;
         } else if (model.IsShapeTensor(tName)) {
            auto data = model.GetShapeTensorValues(tName);
            dim = data[0];
            if (!dim.isParam) {
               value = static_cast<T>(dim.dim);
               ftype = 1;
            } else
               ftype = 2;
         }
         return ftype;
      };

      T start_value;
      T limit_value;
      T delta_value;
      Dim start_dim;
      Dim limit_dim;
      Dim delta_dim;
      int res1 = analyzeInput(fNStart, start_value, start_dim);
      int res2 = analyzeInput(fNLimit, limit_value, limit_dim);
      int res3 = analyzeInput(fNDelta, delta_value, delta_dim);
      if (res1 == 0 || res2 == 0 || res3 == 0) {
         // cannot know at compile time- need to do fully at run time
         //
         fShape = {Dim{"range_size_" + fNStart + "_" + fNLimit}};
         model.AddDynamicTensor(fNOutput, type, fShape);
      } else if (res1 == 1 && res2 == 1 && res3 == 1) {
         size_t number_of_elements = std::max(static_cast<int>(std::ceil((limit_value - start_value) / delta_value )) , 0 );
         fIsOutputConstant = true;

         // compute output
         std::vector<T> output(number_of_elements);
         for (size_t i=0; i<number_of_elements; ++i) {
            output[i] =  start_value + (i * delta_value);
         }
         std::vector<size_t> shape = {number_of_elements};
         model.AddConstantTensor(fNOutput,shape, output.data());
         fShape = ConvertShapeToDim(shape);

          // set the input tensor not writable
         model.SetNotWritableInitializedTensor(fNStart);
         model.SetNotWritableInitializedTensor(fNDelta);
         model.SetNotWritableInitializedTensor(fNLimit);

      } else { // case of a shape tensor
         std::string start = (res1 == 1) ? std::to_string(start_value) : start_dim.GetVal();
         std::string limit = (res2 == 1) ? std::to_string(limit_value) : limit_dim.GetVal();
         std::string delta = (res3 == 1) ? std::to_string(delta_value) : delta_dim.GetVal();
         std::stringstream s;
         if (type == ETensorType::FLOAT ) {
            if (delta_value == 1)
               s <<  "std::max(std::ceil("<< limit << " - " << start << "),0.0f)";
            else
               s <<  "std::max(std::ceil(("<< limit << " - " << start << ")/" << delta << "),0.0f)";
         } else if (type == ETensorType::INT64 ) {
            if (delta == "1") {
               if (start == "0")
                  s <<  limit;
               else
                  s << "std::max((" << limit << " - " << start << "),0L)";
            } else {
               if (start == "0")
                  s <<  "((" << limit << ")/" << delta << ")";
               else
                  s << "std::max((" << limit << " - " << start << ")/"<< delta << "),0L)";
            }
         } else {
            throw
               std::runtime_error("TMVA SOFIE Range Op Input Tensor " + ConvertTypeToString(type) + "is not supported");
         }


         fShape = { Dim {s.str(), static_cast<size_t>(-1)} };
         model.AddDynamicTensor(fNOutput,type, fShape);
      }


      if (model.Verbose()) {
         std::cout << "Range -> output is " << fNOutput << " : " << ConvertShapeToString(fShape);
         if (fIsOutputConstant) std::cout << " : " << ConvertValuesToString(model.GetTensorData<T>(fNOutput));
         std::cout << std::endl;
      }
   }

   std::string Generate(std::string opName) override {

      std::stringstream out;
      out << "\n//------ Range " << opName << "---> " << ConvertDimShapeToString(fShape) << "\n";
      if (fIsOutputConstant) return out.str();

      opName = "op_" + opName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Range operator called to Generate without being initialized first");
      }

      std::string sizeName = fShape[0].param;
      if (sizeName.find("range_size") != std::string::npos)
         sizeName = "static_cast<size_t>(std::max(std::ceil((static_cast<float>(*tensor_" + fNLimit +
                ") - static_cast<float>(*tensor_" + fNStart + ")) / static_cast<float>(*tensor_" + fNDelta + ")), 0.0f))";
      out << SP << "{\n";
      out << SP << SP << "size_t range" << " = " << sizeName << ";\n";
      if (sizeName != fShape[0].param) {
         out << SP << SP << "if ( range > " << "fTensor_" << fNOutput << ".size() ){\n";
         // we should probably resize the tensor here
         out << SP << SP << SP << "throw std::runtime_error(\"wrong size allocated for output of range\");\n";
         out << SP << SP << "}\n";
      }
      out << SP << SP << "for (size_t i = 0; i < range; i++) {\n";
      out << SP << SP << SP << "tensor_" << fNOutput << "[i] = *tensor_" << fNStart << " + i * (*tensor_" << fNDelta << ");\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_RANGE
