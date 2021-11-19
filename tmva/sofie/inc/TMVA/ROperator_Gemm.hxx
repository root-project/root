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
      float fAttrAlpha = 1.0;
      float fAttrBeta = 1.0;
      int_t fAttrTransA = 0;
      int_t fAttrTransB = 0;

      std::string fNA;
      std::string fNB;
      std::string fNC = "";
      std::string fNC2; // bias tensor name after broadcasting
      std::string fNY;
      std::vector<size_t> fShapeA;
      std::vector<size_t> fShapeB;
      std::vector<size_t> fShapeC;
      std::vector<size_t> fShapeY;

      std::string fType;

   public:

      ROperator_Gemm(){}
      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameY):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)) {

         if (std::is_same<T, float>::value) {
            fType = "float";
         }else{
            throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a gemm operator");
         }
      }

      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameC, std::string nameY):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)), fNY(UTILITY::Clean_name(nameY)) {

         if (std::is_same<T, float>::value) {
            fType = "float";
         }else{
            throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a gemm operator");
         }
      }

      std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
         ETensorType out = input[0];
         return {out};
      }

      std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
         if (input.size() > 3) throw std::runtime_error("TMVA SOFIE Gemm Op Shape Inference only need 2 or 3 input tensor");
         for (auto& i: input){
            if (i.size() > 2){
               throw std::runtime_error("TMVA SOFIE Gemm Op Shape Inference only accept input tensor with 2 dimensions");
            }
         }
         std::vector<std::vector<size_t>> ret;
         if (input.size() == 3){
            ret.push_back(input[2]);   //shape of C is shape of Y
            return ret;
         }
         std::vector<size_t> s_a(input[0]);
         std::vector<size_t> s_b(input[1]);
         if (fAttrTransA){
            std::reverse(s_a.begin(), s_a.end());
         }
         if (fAttrTransB){
            std::reverse(s_b.begin(), s_b.end());
         }
         std::vector<size_t> s_y(2);
         s_y[0] = s_a[0];
         s_y[1] = s_b[1];
         ret.push_back(s_y);
         return ret;
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
         fShapeA = model.GetTensorShape(fNA);
         if (fShapeA.size() != 2){
            throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensor" + fNA +
                                     " is not of 2 dimensions: A " +  ConvertShapeToString(fShapeA));
         }
         fShapeB = model.GetTensorShape(fNB);
         if (fShapeB.size() != 2){
            throw std::runtime_error("TMVA SOFIE Gemm Op Input Tensor" + fNB + " is not of 2 dimensions: B " +  ConvertShapeToString(fShapeB));
         }
         fShapeY = ShapeInference({fShapeA, fShapeB})[0];
         if (fNC != ""){
            fShapeC = model.GetTensorShape(fNC);
            fNC2 = fNC;
            bool broadcast_needed = false;

            // broadcast is not needed if fShapeY[0] == 1 i.e. C and Y have same length
            if (ConvertShapeToLength(fShapeC) != ConvertShapeToLength(fShapeY)) {
               broadcast_needed = true;
            }

            // std::cout << "doing broadcast " << broadcast_needed << " use session " << model.UseSession() <<
            //    " shape C " << ConvertShapeToString(fShapeC) << " shape Y " << ConvertShapeToString(fShapeY)
            //                << std::endl;

            if (broadcast_needed) {
               if (!model.UseSession()) {
                  auto original_data = model.GetInitializedTensorData(fNC);
                  if (fType == "float") {

                     std::shared_ptr<void> new_data_ptr(UTILITY::Unidirectional_broadcast<float>(
                                                           static_cast<float *>(original_data.get()), fShapeC, fShapeY),
                                                        std::default_delete<float[]>());

                     model.UpdateInitializedTensor(fNC, model.GetTensorType(fNC), fShapeY, new_data_ptr);
                     fShapeC = fShapeY;
                  }
               } else {
                  // In case of session add broadcasting code in Session constructor and in GenerateInitCode
                  // we need to add a new intermediate tensor for broadcasted bias tensor
                  fNC2 = fNC + "bcast";
                  model.AddIntermediateTensor(fNC2, model.GetTensorType(fNC), fShapeY);
               }
            }
         }




         model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
         model.AddNeededStdLib("algorithm");

      }

      std::string GenerateInitCode()
      {
         std::stringstream out;
         // generate initialization code for broadcasting of bias tensor
         if (fShapeC.size() != fShapeY.size() && fNC != fNC2) {
            // include a separate scope to avoid defining unique operator temp variables
            out << "   {\n";
            out << "      std::vector<size_t> oldShape = " << ConvertShapeToString(fShapeC) << ";\n";
            out << "      std::vector<size_t> newShape = " << ConvertShapeToString(fShapeY) << ";\n";
            std::string original_bias_tensor = "tensor_" + fNC;
            std::string new_bias_tensor = "tensor_" + fNC2;
            out << "      float * newData_ptr = TMVA::Experimental::SOFIE::UTILITY::Unidirectional_broadcast<float>("
                << original_bias_tensor << ", oldShape, newShape);\n";
            int length = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapeY); // output size
            out << "      std::copy(newData_ptr, newData_ptr + " << length << ", " << new_bias_tensor << ");\n";
            out << "      delete [] newData_ptr;\n";
            out << "   }\n";
         }
         return out.str();
      }

      std::string Generate(std::string OpName){
         OpName = "op_" + OpName;

         if (fShapeA.empty() || fShapeB.empty() || fShapeY.empty() || (fNC != "" && fShapeC.empty())) {
            throw std::runtime_error("TMVA SOFIE Gemm Op called to Generate without being initialized first");
         }
         std::stringstream out;
         out << "\n//--------- Gemm\n";
         out << SP << "char " << OpName << "_transA = " << (fAttrTransA ? "\'t\'" : "\'n\'") << ";\n";
         out << SP << "char " << OpName << "_transB = " << (fAttrTransB ? "\'t\'" : "\'n\'") << ";\n";
         int m = (fAttrTransA ? fShapeA[1] : fShapeA[0]);
         int n = (fAttrTransB ? fShapeB[0] : fShapeB[1]);
         int k = (fAttrTransA ? fShapeA[0] : fShapeA[1]);
         out << SP << "int " << OpName << "_m = " << m << ";\n";
         out << SP << "int " << OpName << "_n = " << n << ";\n";
         out << SP << "int " << OpName << "_k = " << k << ";\n";
         out << SP << "float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ";\n";
         out << SP << "float " << OpName << "_beta = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ";\n";
         out << SP << "int " << OpName << "_lda = " << (fAttrTransA ? m : k) << ";\n";
         out << SP << "int " << OpName << "_ldb = " << (fAttrTransB ? k : n) << ";\n";
         if (fNC != ""){
            size_t length = ConvertShapeToLength(fShapeY);
            if (fNC2 == fNC)
               // case broadcasting was not needed or done otside of session
               assert(length == ConvertShapeToLength(fShapeC));
            out << SP << "std::copy(" << "tensor_" << fNC2 << ", " << "tensor_" << fNC2 << " + " << length << ", " << "tensor_" << fNY << ");\n";
         }
         if (fType == "float"){
            out << SP << "BLAS::sgemm_(&" << OpName << "_transB, &" << OpName << "_transA, &" << OpName
             << "_n, &" << OpName << "_m, &" << OpName << "_k, &" << OpName << "_alpha, " << "tensor_" << fNB
             << ", &" << OpName << "_ldb, " << "tensor_" << fNA << ", &" << OpName << "_lda, &" << OpName << "_beta, " << "tensor_" << fNY << ", &"
             << OpName << "_n);\n";
          }

          return out.str();

         }



   };


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GEMM
