#ifndef TMVA_SOFIE_ROPERATOR_Concat
 #define TMVA_SOFIE_ROPERATOR_Concat


 #include "TMVA/SOFIE_common.hxx"
 #include "TMVA/ROperator.hxx"
 #include "TMVA/RModel.hxx"

 #include <sstream>
 #include <algorithm>
 #include <iterator>
 #include <iomanip>
 #include <limits>

 namespace TMVA{
 namespace Experimental{
 namespace SOFIE{

     template <typename T>
     class ROperator_Concat final : public ROperator
     {
     private:
         int fAxis=0;
         int fnewAxis=0;
         std::vector<std::string> fInputs;
         std::string fOutput;
         std::vector<size_t>fOutputShape;
         std::vector<std::vector<size_t>> fInputShapes;

     public:
         ROperator_Concat(){}
         ROperator_Concat(std::vector<std::string> inputs, int axis, int newAxis, std::string output):
         fAxis(axis), fnewAxis(newAxis), fOutput(UTILITY::Clean_name(output)) {
            fInputs.reserve(inputs.size());
            for (auto & name : inputs)
               fInputs.push_back(UTILITY::Clean_name(name));
         }

         std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
             return input;
         }

         // get shape of output given inputs. It is going to be called after initialized
         std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> inputs){
             std::vector<std::vector<size_t>> ret(1);
            // treat negative axis case
            if (fAxis<0) {
               fAxis = inputs[0].size()+fAxis;
            }
            if (fAxis < 0 || fAxis >= (int) inputs[0].size())
               throw std::runtime_error("TMVA SOFIE Concat Op - invalid axis value ");

            int concat_dim=0;
            if(fnewAxis == 0){
               for(size_t i = 0; i < inputs.size(); i++) {
                  if (i > 0 && inputs[i].size() != inputs[i-1].size() )
                  throw std::runtime_error("TMVA SOFIE Concat Op - input tensors have different shapes " +
                     ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i-1]));
                  for (size_t iaxis = 0; iaxis < inputs[i].size(); iaxis++) {
                  if ((int) iaxis == fAxis)
                     concat_dim += inputs[i][iaxis];
                  else
                     if (i> 0 && inputs[i][iaxis] != inputs[i-1][iaxis])
                        throw std::runtime_error("TMVA SOFIE Concat Op - input tensors have wrong shapes " +
                        ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i-1]));
                  }
               }

               // output shape
               ret[0] = inputs[0];
               ret[0][fAxis] = concat_dim;
            }
            std::vector<int> stack;
            if(fnewAxis == 1){
               for(size_t i = 0; i < inputs.size(); i++) {
                  if (i > 0 && inputs[i].size() != inputs[i-1].size() )
                  throw std::runtime_error("TMVA SOFIE Concat Op - input tensors have different shapes " +
                     ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i-1]));
                  for (size_t iaxis = 0; iaxis < inputs[i].size(); iaxis++) {
                     if ((int) iaxis == fAxis)
                        stack.push_back(inputs[i][iaxis]);
                     else
                     if (i> 0 && inputs[i][iaxis] != inputs[i-1][iaxis])
                        throw std::runtime_error("TMVA SOFIE Concat Op - input tensors have wrong shapes " +
                        ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i-1]));
                  }
                  
               }
               for(auto it:stack)
               ret[0].push_back(it);
            }

            return ret;
         }

         void Initialize(RModel &model)
         {
            for (auto &it : fInputs) {
               if (model.CheckIfTensorAlreadyExist(it) == false) {
                  throw std::runtime_error("TMVA SOFIE Concat Op Input Tensor " + it + " is not found in model");
               }
               fInputShapes.push_back(model.GetTensorShape(it));
            }
            fOutputShape = ShapeInference(fInputShapes)[0];
            model.AddIntermediateTensor(fOutput, model.GetTensorType(fInputs[0]), fOutputShape);
         }

         std::string Generate(std::string OpName){
            OpName = "op_"+OpName;
            if(fOutputShape.empty()){
                  throw std::runtime_error("TMVA SOFIE Concat called to Generate without being initialized first");
            }
            std::stringstream out;
            out<<"\n//--------- Concat\n";
            // special case when memory is contigous
            bool hasShapeOnes = true;
            for(int i = 0; i<fAxis; ++i){
               if(fInputShapes[0][i]!=1){
                  hasShapeOnes = false;
                  break;
               }
            }
            if (fAxis == 0 || hasShapeOnes) {
               size_t offset = 0;
               for(size_t i=0; i<fInputs.size(); ++i){
                  out<<SP<<"std::copy(tensor_"<<fInputs[i]<<", tensor_"<<fInputs[i]<<"+"<<ConvertShapeToLength(fInputShapes[i])<<", tensor_"<<fOutput<<"+"<<offset<<");\n";
                  offset+=ConvertShapeToLength(fInputShapes[i]);
               }
            }
            else {

               std::vector<size_t> outStride = UTILITY::ComputeStrideFromShape(fOutputShape);
               std::vector<std::vector<size_t>> inStrides(fInputs.size());
               int idx = 0;
               for ( auto &s : inStrides) {
                  s = UTILITY::ComputeStrideFromShape(fInputShapes[idx]);
                  idx++;
               }
               for (int i = 0; i < fAxis; ++i) {
                  // loop on dimensions
                  out << SP << "for (size_t i" << i << " = 0; i" << i << " < " << fOutputShape[i] << "; ++i" << i <<") {\n";
               }

               out << SP << SP << SP << "int idxOut =";
               for (int k = 0; k < fAxis; k++)
                  out << " + " << outStride[k] << "*i" << k;
               out << ";\n";

               for (size_t j = 0; j < fInputs.size(); j++) {
                  if (j>0)
                  out << SP << SP << SP << "idxOut += " << fInputShapes[j-1][fAxis] << ";\n";
                  out << SP << SP << SP << "int idxIn" << j <<" =";
                  for (int k = 0; k < fAxis; k++)
                     out << " + " << inStrides[j][k] << "*i" << k;
                  out << ";\n";
                  out << SP << SP << SP << "for (size_t iC = 0; iC < " << fInputShapes[j][fAxis] << "; ++iC) {\n";
                  out << SP << SP << SP << SP << "tensor_" << fOutput << "[idxOut+iC] = tensor_" << fInputs[j] << "[idxIn" << j << "+iC];\n";
                  out << SP << SP << SP << "}\n";
               // concatenate the axis values
               }
                for (int i = 0; i < fAxis; ++i) {
                    out << SP << "}\n";
                }
            }

            return out.str();
         }
     };
 }//SOFIE
 }//Experimental
 }//TMVA

 #endif //TMVA_SOFIE_ROPERATOR_CONCAT
