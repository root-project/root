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
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){}

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

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      fShapeInput=model.GetTensorShape(fNInput);

         // Retrieve the data pointer for the repeats tensor
      auto repptr = model.GetInitializedTensorData(fNRepeats);
      // Cast the raw pointer to the appropriate type (size_t*)
      auto repeat_shape = static_cast<size_t*>(repptr.get());

      if (repeat_shape == nullptr) {
        throw std::runtime_error("Failed to retrieve the data for the repeats tensor.");
      }
      // Get the shape of the repeats tensor to determine the number of elements
      auto repeats_shape = model.GetTensorShape(fNRepeats);
      // Ensure the repeats tensor is 1D and get the number of elements
      if (repeats_shape.size() != 1) {
         throw std::runtime_error("Repeats tensor is not 1D.");
      }
      size_t num_elements = repeats_shape[0];
      // Convert the data to a vector
      std::vector<size_t> repeats_vector(repeat_shape, repeat_shape + num_elements);

      fShapeY = ShapeInference({fShapeInput,repeats_vector})[0];

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);
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
      out << "std::vector<int> input_shape = " << ConvertShapeToString(fShapeInput) << ";\n";
      std::vector<size_t> repeats = fShapeY;
      for (size_t i = 0; i < repeats.size(); i++)
         repeats[i] /= fShapeInput[i];

      out << "std::vector<int> repeats = " << ConvertShapeToString(repeats) << ";\n";
      out << "int inputLength = " << ConvertShapeToLength(fShapeInput) << ";\n";
      out << "int s = 1;\n";
      // loop from inverse dim order
      out << "for (int i = " << fShapeInput.size()-1 << "; i >=0; i--) {\n";
      out << SP << "int r = repeats[i];\n";
      // exclude case where repeats=1 apart first one
      out << SP << "if (r == 1 && i < " << fShapeInput.size()-1 <<  ") continue;\n";
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

#if 0 // old inefficient implementation

      out << "std::vector<size_t> input_shape = " << ConvertShapeToString(fShapeInput) << ";\n";
      out << "std::vector<size_t> output_shape = " << ConvertShapeToString(fShapeY) << ";\n";
      out << "std::vector<size_t> indices(input_shape.size(), 0);\n";
      out << "for (size_t i = 0; i < " << output_length << "; ++i) {\n";
      out << SP<<"size_t source_index = 0;\n";
      out << SP<<"size_t stride = 1;\n";
      out << SP<<"for (int j = input_shape.size() - 1; j >= 0; --j) {\n";
      out << SP<<SP<<"source_index += (indices[j] % input_shape[j]) * stride;\n";
      out << SP<<SP<<"stride *= input_shape[j];\n";
      out << SP<<"}\n";
      out << SP<<"tensor_"<<fNY<<"[i] = tensor_"<<fNInput<<"[source_index];\n";
      out << SP<<"for (int j = input_shape.size() - 1; j >= 0; --j) {\n";
      out << SP<<SP<<"if (++indices[j] < output_shape[j]) {\n";
      out << SP<<SP<<SP<<"break;\n";
      out << SP<<SP<<"}\n";
      out << SP<<SP<<"indices[j] = 0;\n";
      out << SP<<"}\n";
      out << "}\n";
      out << "}\n";
#endif
      out << "}\n";  // end of scope
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_Tile