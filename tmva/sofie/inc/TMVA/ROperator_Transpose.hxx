#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cassert>

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
   std::vector<Dim> fShapeData;
   std::vector<Dim> fShapeOutput;

public:

   ROperator_Transpose(){}
   ROperator_Transpose(std::vector<int_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
            fInputTensorNames = { fNData };
            fOutputTensorNames = { fNOutput };
   }

   ROperator_Transpose(std::string nameData, std::string nameOutput):
      fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
         fInputTensorNames = { fNData };
         fOutputTensorNames = { fNOutput };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
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


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         std::cout<<"Input tensor for transpose: "<<fNData<<'\n';
         throw std::runtime_error("TMVA SOFIE Tranpose Op Input Tensor is not found in model");
      }
      fShapeData = model.GetDimTensorShape(fNData);
      if (fAttrPerm.empty()){
         fAttrPerm.reserve(fShapeData.size());
         for (int i = fShapeData.size() - 1; i >= 0; i--){
            fAttrPerm.push_back(i);
         }
      }

      // inference of output shape
      if (fAttrPerm.size() != fShapeData.size() )
         throw std::runtime_error("TMVA SOFIE Tranpose Op - Invalid axes attributes");

      fShapeOutput.resize(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         fShapeOutput[i] = fShapeData[fAttrPerm[i]];
      }

      if (model.IsInitializedTensor(fNData) ) {
         // here we know the shape
         auto shapeX = ConvertShapeToInt(fShapeData);
         auto shapeY = ConvertShapeToInt(fShapeOutput);
         fIsOutputConstant = true;
         // case input is a constant or initialized tensor we perform here the transpose
         auto inStrides = UTILITY::ComputeStrideFromShape(shapeX);
         auto outStrides = UTILITY::ComputeStrideFromShape(shapeY);
         size_t length = ConvertShapeToLength(shapeY);
         auto inputData = static_cast<T*>(model.GetInitializedTensorData(fNData).get());
         size_t dim = fShapeData.size();
         std::vector<size_t> outputIdx(dim);
         std::vector<T> outputData(length);
         for (size_t i = 0; i < length; i++) {
            outputIdx[0] = i / outStrides[0];
            for (size_t j = 1; j < dim; j++) {
               outputIdx[j] = (i % outStrides[j-1]) / outStrides[j];
            }
            // compute input index
            size_t inputIndex = 0;
            for (size_t j = 0; j < dim; j++) {
               // find value in fAtrrPerm corresponding to j
               int k = std::find(fAttrPerm.begin(), fAttrPerm.end(), j) - fAttrPerm.begin();
               inputIndex += outputIdx[k] * inStrides[j];
            }
            outputData[i] = inputData[inputIndex];
         }
         model.AddConstantTensor<T>(fNOutput, shapeY, outputData.data());
         if (model.Verbose()) {
            std::cout << "Transpose: output is a constant tensor " << ConvertShapeToString(shapeY) << " : "
               << ConvertValuesToString(outputData) << std::endl;
         }
      } else {
         model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fShapeOutput);
         if (model.Verbose()) {
            std::cout << "Transpose ---> " << fNOutput << " " <<  ConvertShapeToString(fShapeOutput) << std::endl;
         }
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors
      OpName = "op_" + OpName;
      if (fShapeData.empty() || fShapeOutput.empty()){
         throw std::runtime_error("TMVA SOFIE Transpose Op called to Generate without being initialized first");
      }
      int dim = fShapeData.size();
      auto inStrides = UTILITY::ComputeStrideFromShape(fShapeData);
      auto outStrides = UTILITY::ComputeStrideFromShape(fShapeOutput);
      auto length = ConvertDimShapeToLength(fShapeData);
      //assert (length == outStrides[0]*fShapeOutput[0]);

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
      out << SP << "///------- Transpose operator " << OpName << ConvertDimShapeToString(fShapeData)
                  << " --> " << ConvertDimShapeToString(fShapeOutput) << std::endl;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNOutput << "[id] = tensor_" << fNData << "[ ";
      // compute output j indices
      std::vector<std::string> i_out(dim);
      for (int k =0; k < dim; k++){
         if (k == 0)
            i_out[k] = "id";
         else
            i_out[k] = "(id % (" + outStrides[k-1].GetVal() + "))";
         if (k < dim-1)
            i_out[k] += " / (" + outStrides[k].GetVal() + ")";
      }
      // use now them for input tensors
      // need to invert the fAttrPerm[k]
      for (int k =0; k < dim; k++){
         // find value in fAtrrPerm corresponding to k
         int l = std::find(fAttrPerm.begin(), fAttrPerm.end(), k) - fAttrPerm.begin();
         assert(l >= 0 && l < dim);
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
