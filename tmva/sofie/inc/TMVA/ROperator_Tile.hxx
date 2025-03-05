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

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
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

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
      }

      //size_t input_length = ConvertShapeToLength(fShapeInput);
      //size_t output_length = ConvertShapeToLength(fShapeY);


      std::stringstream out;
      std::string input = "tensor_" + fNInput;
      std::string output = "tensor_" + fNY;
      out << "///-------- Tile operator\n";
      out << "{\n"; // add scope to re-use same names
      out << "const int input_shape[" << fShapeInput.size() << "] = " << ConvertShapeToString(fShapeInput) << ";\n";

      out << "int inputLength = " << ConvertShapeToLength(fShapeInput) << ";\n";
      out << "int s = 1;\n";
      // loop from inverse dim order
      out << "for (int i = " << fShapeInput.size()-1 << "; i >=0; i--) {\n";
      out << SP << "int r = tensor_" << fNRepeats << "[i];\n";
      // we cannot exclude case where repeats=1 since we need offset
      //out << SP << "if (r == 1 && i < " << fShapeInput.size()-1 <<  ") continue;\n";
      out << SP << "int i_offset = 0, o_offset = 0;\n";
      out << SP << "s = s * input_shape[i];\n";
      // case we have first copy
      out << SP << "if (i == " << fShapeInput.size()-1 <<  ") {\n";
      out << SP << SP <<  "for (int j = 0; j < inputLength/s ; j++) {\n";
      out << SP << SP << SP << "for (int k = 0; k < r ; k++) {\n";
      out << SP << SP << SP << SP << "std::copy(" << input << "+ i_offset, "
                                    << input << "+ i_offset + s, " << output << "+ o_offset);\n";
      out << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << "}\n"; // end k loop
      out << SP << SP << SP << "i_offset += s;\n";
      out << SP << SP << "}\n"; // end j loop
      out << SP << "} else {\n";  // second copy we do from output to output
      // and we need to loop on j from reverse order to avoir re-writing in output tensor
      out << SP << SP << "for (int j = inputLength/s - 1 ; j>=0; j--) {\n";
      out << SP << SP << SP << "o_offset = j*s*r;\n";
      out << SP << SP << SP << "i_offset = j*s;\n";
      out << SP << SP << SP << "for (int k = 0; k < r ; k++) {\n";
      out << SP << SP << SP << SP << "std::copy(" << output << "+ i_offset, "
                                    << output << "+ i_offset + s, " << output << "+ o_offset);\n";
      out << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << "}\n"; // end k loop
      out << SP << SP << "}\n"; // end j loop
      out << SP << "}\n"; // end if
      out << SP << "s *= r;\n";
      out << SP << "inputLength *= r;\n";
      out << "}\n"; // end i loop
      out << "}\n";  // end of scope
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_Tile
