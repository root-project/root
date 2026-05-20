#ifndef TMVA_SOFIE_ROPERATOR_ScatterND
#define TMVA_SOFIE_ROPERATOR_ScatterND

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator_ScatterND final : public ROperator
{
private:


   std::string fNX;
   std::string fNI;
   std::string fNU;
   std::string fNY;
   std::string fReduction;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeI;
   std::vector<Dim> fShapeY;


   std::vector<int64_t> fIndices;  // indices vector in case they are known at initialization

   std::string fType;


public:
   ROperator_ScatterND(){}
   ROperator_ScatterND(const std::string & nameX, const std::string & nameI, const std::string & nameU, const std::string & nameY,
                        std::string reduction):
      fNX(UTILITY::Clean_name(nameX)), fNI(UTILITY::Clean_name(nameI)), fNU(UTILITY::Clean_name(nameU)),
      fNY(UTILITY::Clean_name(nameY)), fReduction(reduction)
   {
      fInputTensorNames = { fNX, fNI, fNU };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {

       // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error(std::string("TMVA SOFIE ScatterND Op Input Tensor ") + fNX + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNI)) {
         throw std::runtime_error(std::string("TMVA SOFIE ScatterND Op Input Tensor ") + fNI + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNU)) {
         throw std::runtime_error(std::string("TMVA SOFIE ScatterND Op Input Tensor ") + fNU + "is not found in model");
      }
      //tbd check for constant tensors

      fShapeX = model.GetDimTensorShape(fNX);
      fShapeI = model.GetDimTensorShape(fNI);
      auto shapeU = model.GetDimTensorShape(fNU);

      //  Validate inputs if fShapeI last is not dynamic

      //if (!model.IsDynamicTensor(fNI)) {
      const size_t r = fShapeX.size();       // rank of data
      const size_t q = fShapeI.size();    // rank of indices
      if (!(fShapeI.back().isParam) ) {
         const size_t k = fShapeI.back().dim;             // index depth

         if (k > r)
            throw std::invalid_argument(
               "ScatterND: last dim of indices (" + std::to_string(k) +
               ") must be <= rank of data (" + std::to_string(r) + ")");

         // Expected updates rank = q - 1 + r - k
         int64_t expected_updates_rank = q - 1 + r - k;
         if ((int64_t) shapeU.size() != expected_updates_rank)
            throw std::invalid_argument("ScatterND: updates rank mismatch");
      } else {
         //  Assumption is that last dimension of index shape is known (is not dynamic)
         throw std::runtime_error("TMVA SOFIE ScatterND : Index_shape(-1) is not known. This case is not supported");
      }

      // output shape is equal to input shape
      fShapeY = fShapeX;

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()) {
         std::cout << "ScatterElements: input: " << ConvertDimShapeToString(fShapeX)
                                                << " indices " << ConvertDimShapeToString(fShapeI)
                                                << " update " <<  ConvertDimShapeToString(shapeU);
         std::cout << "\t----> " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         // no code to generate here for constant output. Tensor output is defined in Session constructor
         return "//---------------------------------------\n";
      }
      opName = "op_" + opName;
      std::stringstream out;
      out << "//--------- ScatterND " << opName << " --> " << ConvertDimShapeToString(fShapeY) << "\n";

      size_t r = fShapeX.size();

      // Strides
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto stridesI = UTILITY::ComputeStrideFromShape(fShapeI);

      // case input_index_shape == rank of input
      size_t k = fShapeI.back().dim;

      // Total number of index tuples = product of indices dims except last
      std::vector<Dim> shapeIndFirst(fShapeI.begin(), fShapeI.begin()+ fShapeI.size()-1);
      auto num_index_tuples = ConvertDimShapeToLength(shapeIndFirst);

      //slice size (is product of input from k to r)
      std::vector<Dim> shapeSlice(fShapeX.begin()+k, fShapeX.end());
      auto slice_size = ConvertDimShapeToLength(shapeSlice);

      auto data_length = ConvertDimShapeToLength(fShapeX);

      //step1: input->output
      out << SP << "// Step 1: copy input data to output\n";
      out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << data_length << ", tensor_" << fNY << ");\n";

      // Step 2: Emit strides as a static constexpr array
      out << SP << "// Step 2: data strides (row-major)\n";
      //to do: use static constexpr for defined strides
      out << SP << "size_t " << opName << "_data_strides[" << r << "] = {";
      for (size_t i = 0; i < r; ++i)
         out << stridesX[i] << (i + 1 < r ? ", " : "");
      out << "};\n\n";

      // Step 3: Scatter loop
      out << SP << "// Step 3: scatter updates into output\n";
      out << SP << "for (int64_t idx = 0; idx < " << num_index_tuples << "; idx++) {\n";

      // Resolve flat data offset from k-dimensional index tuple
      out << SP << SP << "int64_t data_offset = 0;\n";
      for (size_t dim = 0; dim < k; ++dim) {
         out << SP << SP << "{\n";
         out << SP << SP << SP << "int64_t coord = tensor_" << fNI
             << "[idx * " << k << " + " << dim << "];\n";
         // Support negative indices
         out << SP << SP << SP << "if (coord < 0) coord += " << fShapeX[dim] << ";\n";
         out << SP << SP << SP << "data_offset += coord * "
               << opName << "_data_strides[" << dim << "];\n";
         out << SP << SP << "}\n";
      }

      // Apply updates with reduction
      out << SP << SP << "for (int64_t s = 0; s < " << slice_size << "; s++) {\n";
      out << SP << SP << SP << "auto upd = tensor_" << fNU
         << "[idx * " << slice_size << " + s];\n";

      if (fReduction.empty() || fReduction == "none") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] = upd;\n";
      } else if (fReduction == "add") {
         out << SP << SP << SP << "tensor_" << fNY<< "[data_offset + s] += upd;\n";
      } else if (fReduction == "mul") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] *= upd;\n";
      } else if (fReduction == "min") {
         out << SP << SP << SP << "tensor_" << fNY<< "[data_offset + s] = "
               << "std::min(tensor_" << fNY << "[data_offset + s], upd);\n";
      } else if (fReduction == "max") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] = "
            << "std::max(tensor_" << fNY << "[data_offset + s], upd);\n";
      } else {
         throw std::runtime_error(
            "TMVA SOFIE ScatterND: unsupported reduction '" + fReduction + "'");
      }

      out << SP << SP << "}\n";  // end slice loop
      out << SP << "}\n";        // end index tuple loop

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RELU
