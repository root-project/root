#ifndef TMVA_SOFIE_ROPERATOR_GEMM
#define TMVA_SOFIE_ROPERATOR_GEMM


#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{


   template <typename T>
   class ROperator_Gemm final : public ROperator
   {

   private:
      bool fIsDynamic = false;

      float fAttrAlpha = 1.0;
      float fAttrBeta = 1.0;
      int_t fAttrTransA = 0;
      int_t fAttrTransB = 0;

      std::string fNA;
      std::string fNB;
      std::string fNC = "";
      std::string fNC2; // bias tensor name after broadcasting
      std::string fNY;
      std::string fType;
      std::vector<Dim> fShapeA;
      std::vector<Dim> fShapeB;
      std::vector<size_t> fShapeC;
      std::vector<Dim> fShapeY;

   public:

      ROperator_Gemm(){}
      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameY):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY))
      {
         fType = "float";
         static_assert(std::is_same_v<T, float>,
                  "TMVA::SOFIE - Unsupported type parsing a Gemm operator");
      }

      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameC, std::string nameY):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)), fNY(UTILITY::Clean_name(nameY))
      {
         fType = "float";
         static_assert(std::is_same_v<T, float>,
                  "TMVA::SOFIE - Unsupported type parsing a Gemm operator");
      }

      std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
         ETensorType out = input[0];
         return {out};
      }

      template <typename U>
      std::vector<std::vector<U>> DoShapeInference(const std::vector<std::vector<U>> & input){
         if (input.size() > 3) throw std::runtime_error("TMVA SOFIE Gemm Op Shape Inference only need 2 or 3 input tensor");
         // accept tensor with input dimensions > 2
         // example: A = (d1,d2,...,N1,N2)  B = (d1,d2,...,N2,N3)    --> Y = (d1,d2,..,N1,N3)
         for (auto& i: input){
            if (i.size() < 2){
               throw std::runtime_error("TMVA SOFIE Gemm Op Shape Inference only accept input tensor with >=2 dimensions");
            }
         }

         std::vector<std::vector<U>> ret;
         // when there are 3 inputs shape of Y is the one of C
         if (input.size() == 3){
            ret.push_back(input[2]);   //shape of C is shape of Y
            return ret;
         }
         // ioffset cannot be less than 2
         int ioffset = input[0].size()-2;  // in case of tensors with dim > 2

         std::vector<U> s_a(input[0].begin() + ioffset, input[0].begin() + ioffset + 2);
         std::vector<U> s_b(input[1].begin() + ioffset, input[1].begin() + ioffset + 2);
         // reverse in case of transpose
         if (fAttrTransA){
            std::reverse(s_a.begin(), s_a.end());
         }
         if (fAttrTransB){
            std::reverse(s_b.begin(), s_b.end());
         }
         std::vector<U> s_y;
         s_y.reserve(input[0].size());
         if (input[0].size() > 2 && input[1].size() == input[0].size()) {
            // in case of dim > 2 first dimensions are equal to input ones
            for (size_t i = 0; i < input[0].size()-2; i++)
               s_y.push_back(input[0][i]);
         }

         s_y.push_back(s_a[0]);
         s_y.push_back(s_b[1]);
         ret.push_back(s_y);
         return ret;
      }

      std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
         return DoShapeInference<size_t>(input);
      }
      std::vector<std::vector<Dim>> DynamicShapeInference(const std::vector<std::vector<Dim>> & input){
         return DoShapeInference<Dim>(input);
      }



      void Initialize(RModel& model){
         //TODO: propagate A or B as specified by ONNX standard

         if ((model.CheckIfTensorAlreadyExist(fNA) == false) || (model.CheckIfTensorAlreadyExist(fNB) == false) ){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensor " + fNA + " or " + fNB + " is not found in model");
         }
         if (fNC != ""){
            if (model.CheckIfTensorAlreadyExist(fNC) == false){   //input must be a graph input, or already initialized intermediate tensor
               throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensor" + fNC + " is not found in model");
            }
         }
         if (model.IsDynamicTensor(fNA) || model.IsInputTensor(fNA) ) {
            fShapeA = model.GetDynamicTensorShape(fNA);
            fIsDynamic = true;
         } else {
            auto shapeA_int = model.GetTensorShape(fNA);
            fShapeA = ConvertShapeToDim(shapeA_int);
         }
         // case A is of dim1 we prepend a 1 but we need to remove later
         bool appendOne = false;
         if (fShapeA.size() == 1) {
            fShapeA.insert(fShapeA.begin(), Dim(1));
            appendOne = true;
         }

         if (model.IsDynamicTensor(fNB) || model.IsInputTensor(fNB)) {
            fShapeB = model.GetDynamicTensorShape(fNB);
            fIsDynamic = true;
         }
         else {
            auto shapeB_int = model.GetTensorShape(fNB);
            fShapeB = ConvertShapeToDim(shapeB_int);
         }
         // assume if not shape is 2 that extra values are 1.
         // we need to implement MatMul case where we stack matrices (see numpy.matmul)

         if (fShapeA.size() != fShapeB.size())
              throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensors have not compatible shapes. A " +
               ConvertDynamicShapeToString(fShapeA) + " B " + ConvertDynamicShapeToString(fShapeB) );

         fShapeY = DynamicShapeInference({fShapeA, fShapeB})[0];
         std::vector<size_t> shapeY;
         if (!fIsDynamic) {
            shapeY = ConvertShapeToInt(fShapeY);
            if (shapeY.empty()) {
               throw std::runtime_error("TMVA SOFIE Gemm Op " + fNY + " has invalid shape" + ConvertDynamicShapeToString(fShapeY));
            }
         }

         // bias is normally not dynamic (not support it for time being)
         if (fNC != ""){
            // normally bias is fixed and not dynamic
            if (model.IsDynamicTensor(fNC)) {
               throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensor" + fNC + " is dynamic and is not supported");
            }
            fShapeC = model.GetTensorShape(fNC);
            fNC2 = fNC;
            bool broadcast_needed = !UTILITY::AreSameShape(fShapeC, fShapeY);

            // For Gemm broadcasting is not needed if fShapeY[0] == 1 i.e. C and Y have same length
            //if (fShapeY[0] == 1 && ConvertShapeToLength(fShapeC) != ConvertShapeToLength(fShapeY)) {
            //   broadcast_needed = false;
            //}


            if (broadcast_needed) {
               if (!model.UseSession()) {
                  // without session dynamic tensors not supported in Gemm
                  if (fIsDynamic) {
                      throw std::runtime_error("TMVA SOFIE Gemm Op:  dynamic tensors not supported without a session");
                  }
                  auto original_data = model.GetInitializedTensorData(fNC);
                  auto targetShape = UTILITY::UnidirectionalBroadcastShape(fShapeC, shapeY);
                  if (fType == "float") {
                     std::shared_ptr<void> new_data_ptr(UTILITY::UnidirectionalBroadcast<float>(
                        static_cast<float *>(original_data.get()), fShapeC, targetShape),
                        std::default_delete<float[]>());

                     model.UpdateInitializedTensor(fNC, model.GetTensorType(fNC), shapeY, new_data_ptr);
                     fShapeC = shapeY;
                  }
               } else {
                  // In case of session add broadcasting code in Session constructor and in GenerateInitCode
                  // we need to add a new intermediate tensor for broadcasted bias tensor
                  fNC2 = fNC + "bcast";
                  if (!fIsDynamic) {
                     model.AddIntermediateTensor(fNC2, model.GetTensorType(fNC), shapeY);
                  }
                  else
                     model.AddDynamicTensor(fNC2,model.GetTensorType(fNC), fShapeY);
               }
            }
         }

         if (appendOne) {
            // remove appended value of 1
            if (fIsDynamic)
               fShapeY.erase(fShapeY.begin());
            else
               shapeY.erase(shapeY.begin());
         }

         if (!fIsDynamic)
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), shapeY);
         else
            model.AddDynamicTensor(fNY, model.GetTensorType(fNA), fShapeY);

         if (model.Verbose()){
            std::cout << "Gemm (or MatMul) " << " ---> " << fNY << " shape ";
            if (fIsDynamic)
               std::cout << ConvertDynamicShapeToString(fShapeY) << std::endl;
            else
               std::cout << ConvertShapeToString(shapeY) << std::endl;
         }

         model.AddNeededStdLib("algorithm");

      }

      std::string GenerateInitCode()
      {
         std::stringstream out;
         // generate initialization code for broadcasting of bias tensor
         if (fShapeC.size() != fShapeY.size() && fNC != fNC2) {
            // we broadcast here always C in Y output, so target shape is the one of Y
            // no need to call UTILITY::UnidirectionalBroadcastShape.
            // here in case of parametric shape we need to assume that the parameters will be defined in the initialization code.
            auto targetShape = fShapeY;
            // include a separate scope to avoid defining unique operator temp variables
            out << "//--- broadcast bias tensor " << fNC << "for Gemm op\n";
            out << SP << "{\n";
            out << "      float * data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_"
               << fNC << "," << ConvertShapeToString(fShapeC) << ", " << ConvertDynamicShapeToString(fShapeY) << ");\n";
            auto length = TMVA::Experimental::SOFIE::ConvertDynamicShapeToLength(fShapeY); // output size
            out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNC2 << ");\n";
            out << SP << SP << "delete [] data;\n";
            out << SP << "}\n";
         }
         return out.str();
      }

      std::string Generate(std::string opName){
         opName = "op_" + opName;

         if (fShapeA.empty() || fShapeB.empty() || fShapeY.empty() || (fNC != "" && fShapeC.empty())) {
            throw std::runtime_error("TMVA SOFIE Gemm Op called to Generate without being initialized first");
         }
         std::stringstream out;
         out << "\n//--------- Gemm\n";
         out << SP << "char " << opName << "_transA = " << (fAttrTransA ? "\'t\'" : "\'n\'") << ";\n";
         out << SP << "char " << opName << "_transB = " << (fAttrTransB ? "\'t\'" : "\'n\'") << ";\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         if (dimA != dimB || dimA != dimY) {
             throw std::runtime_error("TMVA SOFIE Gemm(MatMul) has invalid shape for inputs or output");
         }
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         std::vector<Dim> sY = {fShapeY[dimY-2], fShapeY[dimY-1]};
         std::vector<Dim> sA;
         for (int64_t i = 0; i < dimY-2; i++) {
            sA.push_back(fShapeY[i]);
         }
         auto lengthGemm = ConvertDynamicShapeToLength(sY); // size of the Gemm operation
         auto lengthExtra = ConvertDynamicShapeToLength(sA); // extra length in case input tensors are of dim>2 (MatMul)

         out << SP << "int " << opName << "_m = " << m << ";\n";
         out << SP << "int " << opName << "_n = " << n << ";\n";
         out << SP << "int " << opName << "_k = " << k << ";\n";
         out << SP << "float " << opName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ";\n";
         out << SP << "float " << opName << "_beta = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ";\n";
         out << SP << "int " << opName << "_lda = " << (fAttrTransA ? m : k) << ";\n";
         out << SP << "int " << opName << "_ldb = " << (fAttrTransB ? k : n) << ";\n";

         // case bias is present
         if (!fNC.empty()){
            if (fNC2 == fNC) {
               // add a check in case broadcasting was not needed or done outside of session
               // C should have smaller dimension of Y
               if (!fIsDynamic) {
                  if (std::stoi(lengthGemm) != static_cast<int>(ConvertShapeToLength(fShapeC)))
                     throw std::runtime_error("TMVA SOFIE Gemm Op " + opName + " Bias tensor has not correct size "
                            + ConvertShapeToString(fShapeC) + " output length " + lengthGemm);
               } else {
                  // add a dynamic check (C should not be a dynamic tensor)
                  out << SP << "assert(" << lengthGemm << " != " <<  ConvertShapeToLength(fShapeC) << ");\n";
               }
            }
         } else {
            //in this case fAttrBeta needs to be equal to zero otherwise second time we run we will use
            // the previous result
            if (fAttrBeta != 0) {
               throw std::runtime_error("TMVA SOFIE Gemm Op " + opName + " Bias tensor is not present but beta value in Gemm is not zero");
            }
         }

         // include MatMul case where we stack the Gemm operations
         // exclude case where we have only 1's in the additional dims
         bool doStackMul = dimY > 2 && ( fIsDynamic  || std::stoi(lengthExtra) > 1);
         if (doStackMul) {
            out << SP << "size_t " << opName << "_yoffset = 0;\n"; // needed if we stack the gemm operations
            out << SP << "for (int i = 0; i < " << lengthExtra << "; i++){\n";
            out << SP;
         }
         // in the case of bias
         if (!fNC.empty()){
            out << SP << "std::copy(" << "tensor_" << fNC2 << ", " << "tensor_" << fNC2 << " + " << lengthGemm << ", "
               << "tensor_" << fNY;
            if (doStackMul) out << " + " << opName << "_yoffset";
            out << ");\n";
         }


         if (fType == "float"){

            out << SP << "BLAS::sgemm_(&" << opName << "_transB, &" << opName << "_transA, &" << opName
             << "_n, &" << opName << "_m, &" << opName << "_k, &" << opName << "_alpha, " << "tensor_" << fNB
             << ", &" << opName << "_ldb, " << "tensor_" << fNA << ", &" << opName << "_lda, &" << opName << "_beta, "
             << "tensor_" << fNY;
             if (doStackMul) out << " + " << opName << "_yoffset";
             out << ", &" << opName << "_n);\n";
         }

         if (doStackMul) {
            out << SP << SP <<  opName << "_yoffset += " << lengthGemm << ";\n";
            out << "}\n"; // end of loop on the stacked multiplications
         }

         return out.str();
      }

      std::vector<std::string> GetBlasRoutines() { return { std::string("Gemm"), std::string("Gemv") }; }

   };


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GEMM
