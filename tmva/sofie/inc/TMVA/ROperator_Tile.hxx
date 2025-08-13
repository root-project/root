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
   std::vector<size_t>fShapeInput;
   std::vector<size_t> fShapeY;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNRepeats, fNInput };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      std::vector<size_t> ret = input[0];

      for(size_t i=0; i < input[1].size(); i++) {
            ret[i]=ret[i]*input[1][i];
      }
      return {ret};
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      fShapeInput=model.GetTensorShape(fNInput);

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


      fShapeY = ShapeInference({fShapeInput,repeats_vector})[0];

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout <<  "Tile: " << fNInput << " " << ConvertShapeToString(fShapeInput) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
            << " given repeats " << ConvertShapeToString(repeats_vector) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
      }

      //size_t input_length = ConvertShapeToLength(fShapeInput);
      //size_t output_length = ConvertShapeToLength(fShapeY);


      std::stringstream out;
      out << "///-------- Tile operator (Optimized Direct Mapping)\n";
      out << "{\n";

      const int rank = fShapeInput.size();
      
      out << SP << "const int input_shape[" << rank << "] = " << ConvertShapeToString(fShapeInput) << ";\n";
      out << SP << "const int output_shape[" << rank << "] = " << ConvertShapeToString(fShapeY) << ";\n\n";

      // Pre-calculating strides for both input and output tensors to find element positions.
      out << SP << "int input_strides[" << rank << "];\n";
      out << SP << "int output_strides[" << rank << "];\n";
      out << SP << "input_strides[" << rank - 1 << "] = 1;\n";
      out << SP << "output_strides[" << rank - 1 << "] = 1;\n";
      out << SP << "for (int i = " << rank - 2 << "; i >= 0; --i) {\n";
      out << SP << SP << "input_strides[i] = input_strides[i+1] * input_shape[i+1];\n";
      out << SP << SP << "output_strides[i] = output_strides[i+1] * output_shape[i+1];\n";
      out << SP << "}\n\n";

      out << SP << "const int output_size = " << ConvertShapeToLength(fShapeY) << ";\n";

      // Main loop
      out << SP << "for (int out_idx = 0; out_idx < output_size; ++out_idx) {\n";
      out << SP << SP << "int current_idx = out_idx;\n";
      out << SP << SP << "int in_idx = 0;\n";
      
      // For each output element, calculating the corresponding input element's index.
      out << SP << SP << "for (int i = 0; i < " << rank << "; ++i) {\n";
      out << SP << SP << SP << "const int out_coord = current_idx / output_strides[i];\n";
      out << SP << SP << SP << "const int in_coord = out_coord % input_shape[i];\n";
      out << SP << SP << SP << "in_idx += in_coord * input_strides[i];\n";
      out << SP << SP << SP << "current_idx %= output_strides[i];\n";
      out << SP << SP << "}\n";
      
      out << SP << SP << "tensor_" << fNY << "[out_idx] = tensor_" << fNInput << "[in_idx];\n";
      out << SP << "}\n"; // End of main loop
      
      out << "}\n"; // End of scope
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_Tile
