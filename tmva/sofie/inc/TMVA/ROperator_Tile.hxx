#ifndef TMVA_SOFIE_ROPERATOR_Tile
#define TMVA_SOFIE_ROPERATOR_Tile

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator
{

private:

   std::string fNRepeats;
   std::string fNInput;
   std::string fNY;
   std::vector<Dim>fShapeInput;
   std::vector<Dim> fShapeY;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){
         // the repeats tensor is only used at generation time, so it is not a runtime input
         fInputTensorNames = { fNInput };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<Dim> DoShapeInference(const std::vector<Dim> & input, const std::vector<size_t> repeat)  {
      std::vector<Dim> ret = input;
      for(size_t i=0; i < repeat.size(); i++) {
         if (repeat[i] != 1) {
            if (ret[i].isParam) {
               // parenthesize in case the dimension is a compound expression (e.g. "bsize + 1")
               ret[i] = Dim{ std::string("(" + ret[i].GetVal() + ")*" + std::to_string(repeat[i])), static_cast<size_t>(-1) };
            } else {
               ret[i]=Dim { ret[i].dim *repeat[i] };
            }
         }
      }
      return ret;
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      fShapeInput=model.GetDimTensorShape(fNInput);

      // if repeats vector is not initialized we cannot deduce shape of output
      // not support for time being this case
      if (!model.IsInitializedTensor(fNRepeats)) {
         throw std::runtime_error("TMVA SOFIE Tile Op: non-initialized repeats input is not supported");
      }

      // Retrieve the data pointer for the repeats tensor
      auto repptr = model.GetInitializedTensorData(fNRepeats);
      // Cast the raw pointer to the appropriate type (size_t*)
      auto repeats_data = static_cast<int64_t*>(repptr.get());
      if (repeats_data == nullptr) {
        throw std::runtime_error("Failed to retrieve the data for the repeats tensor.");
      }
      // Get the shape of the repeats tensor to determine the number of elements
      auto repeats_shape = model.GetTensorShape(fNRepeats);
      // Ensure the repeats tensor is 1D and get the number of elements
      if (repeats_shape.size() != 1) {
         throw std::runtime_error("Repeats tensor is not 1D.");
      }
      size_t num_elements = repeats_shape[0];
      // Convert the data to a vector of size_t
      std::vector<size_t> repeats_vector(num_elements);
      std::copy(repeats_data, repeats_data + num_elements, repeats_vector.begin());


      fShapeY = DoShapeInference(fShapeInput,repeats_vector);

      // the repeats are baked into the generated code, so the tensor is not
      // needed at runtime and must not be written in the weight file
      model.SetNotWritableInitializedTensor(fNRepeats);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout <<  "Tile: " << fNInput << " " << ConvertDimShapeToString(fShapeInput) << " -> " << fNY << " with shape " << ConvertDimShapeToString(fShapeY)
            << " given repeats " << ConvertShapeToString(repeats_vector) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
      }

      std::stringstream out;
      out << "///-------- Tile operator " << OpName << "\n";
      out << "{\n";

      const int rank = fShapeInput.size();

      // shapes can contain dynamic (parametric) dimensions, so they are emitted
      // as expressions evaluated at runtime in the generated code
      out << SP << "const size_t input_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeInput) << ";\n";
      out << SP << "const size_t output_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeY) << ";\n\n";

      // Pre-calculating the input strides to find element positions (the output
      // index just advances sequentially in the loop nest below).
      out << SP << "size_t input_strides[" << rank << "];\n";
      out << SP << "input_strides[" << rank - 1 << "] = 1;\n";
      out << SP << "for (int i = " << rank - 2 << "; i >= 0; --i) {\n";
      out << SP << SP << "input_strides[i] = input_strides[i+1] * input_shape[i+1];\n";
      out << SP << "}\n\n";

      // One loop per output axis: o<i> is the output coordinate and ic<i> the
      // corresponding input coordinate, kept in sync via a wrap-around counter
      // so no division or modulo is needed per element.
      out << SP << "size_t out_idx = 0;\n";
      std::string indent = SP;
      for (int i = 0; i < rank; ++i) {
         out << indent << "for (size_t o" << i << " = 0, ic" << i << " = 0; o" << i
             << " < output_shape[" << i << "]; ++o" << i << ") {\n";
         indent += SP;
         out << indent << "const size_t in_off" << i << " = "
             << (i == 0 ? std::string() : "in_off" + std::to_string(i - 1) + " + ")
             << "ic" << i << " * input_strides[" << i << "];\n";
      }
      out << indent << "tensor_" << fNY << "[out_idx++] = tensor_" << fNInput << "[in_off" << rank - 1 << "];\n";
      for (int i = rank - 1; i >= 0; --i) {
         out << indent << "if (++ic" << i << " == input_shape[" << i << "]) ic" << i << " = 0;\n";
         indent.resize(indent.size() - SP.size());
         out << indent << "}\n";
      }

      out << "}\n"; // End of scope
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_Tile
