#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{




template <typename T>
class ROperator_Transpose final : public ROperator
{

private:
   std::vector<int_t> fAttrPerm;

   std::string fNData;
   std::string fNOutput;
   std::vector<size_t> fShapeData;
   std::vector<size_t> fShapeOutput;

public:

   ROperator_Transpose(){}
   ROperator_Transpose(std::vector<int_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
   }

   ROperator_Transpose(std::string nameData, std::string nameOutput):
      fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      if (input.size() > 1) throw std::runtime_error("TMVA SOFIE Tranpose Op Shape Inference only need 1 input tensor");
      auto& data = input[0];
      if (fAttrPerm.size() != data.size() )
         throw std::runtime_error("TMVA SOFIE Tranpose Op - Invalid axes attributes");

      std::vector<size_t> output_shape(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         output_shape[i] = data[fAttrPerm[i]];
      }
      std::vector<std::vector<size_t>> ret;
      ret.push_back(output_shape);
      return ret;
   }


   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Tranpose Op Input Tensor is not found in model");
      }
      fShapeData = model.GetTensorShape(fNData);
      if (fAttrPerm.empty()){
         fAttrPerm.reserve(fShapeData.size());
         for (int i = fShapeData.size() - 1; i >= 0; i--){
            fAttrPerm.push_back(i);
         }
      }
      std::vector<std::vector<size_t>> inputs = { fShapeData };
      fShapeOutput = ShapeInference(inputs).front();
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeData.empty() || fShapeOutput.empty()){
         throw std::runtime_error("TMVA SOFIE Transpose Op called to Generate without being initialized first");
      }
      int dim = fShapeData.size();
      auto inStrides = UTILITY::ComputeStrideFromShape(fShapeData);
      auto outStrides = UTILITY::ComputeStrideFromShape(fShapeOutput);
      size_t length = inStrides[0]*fShapeData[0];  // total tensor size
      assert (length == outStrides[0]*fShapeOutput[0]);

      std::stringstream out;
      // Implement transpose operator using consecutive read inputs.
      // But
      // tensorOut[id] = tensorInput[ inStrides[0]*i0 + inStrides[1]*i1 + inStrides[2]*i2 + ...]
      // now if (j0,j1,j2) are the output indices
      // j0 =  id / outStrides[0]
      // j1 =  (id % outStrides[0])/outStrides[1]
      // j2 =  (id % outStrides[1])/outStrides[2]
      //......
      // and we have j_k = i_fAttrPerm[k]
      // since we are using consecutive writes we should find the inverse of fAttrPerm
      out << SP << "///------- Transpose operator\n" << std::endl;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNOutput << "[id] = tensor_" << fNData << "[ ";
      // compute output j indices
      std::vector<std::string> i_out(dim);
      for (int k =0; k < dim; k++){
         if (k == 0)
            i_out[k] = "id";
         else
            i_out[k] = "(id % " + std::to_string(outStrides[k-1]) + ")";
         if (k < dim-1)
            i_out[k] += " / " + std::to_string(outStrides[k]);
      }
      // use now them for input tensors
      // need to invert the fAttrPerm[k]
      for (int k =0; k < dim; k++){
         // find value in fAtrrPerm corresponding to k
         int l = std::find(fAttrPerm.begin(), fAttrPerm.end(), k) - fAttrPerm.begin();
         assert(l > 0 && l < dim);
         out << "( " << i_out[l] << " )";
         if (k < dim-1) {
            out << " * " << inStrides[k];
            out << " + ";
         }
      }
      out << "];\n";
      out << SP << "}\n";
      return out.str();
   }


};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_TRANSPOSE
