#include "TMVA/SOFIE_common.hxx"
#include<cctype>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

std::vector<Dim> ConvertShapeToDim(std::vector<size_t> shape){
   std::vector<Dim> fshape(shape.size());
   for (size_t i =0; i < shape.size(); i++){
      fshape[i].dim = shape[i];
   }
   return fshape;
}

std::size_t ConvertShapeToLength(std::vector<size_t> shape){
   std::size_t fLength = 1;
   for (auto& dim: shape) fLength *= dim;
   return fLength;
}

std::string ConvertTypeToString(ETensorType type){
   switch(type){
      case ETensorType::FLOAT : {
         return "float";
      }
      case ETensorType::INT16 : {
         return "int16_t";
      }
      case ETensorType::INT32 : {
         return "int32_t";
      }
      case ETensorType::INT64 : {
         return "int64_t";
      }
      case ETensorType::UINT16 : {
         return "uint16_t";
      }
      case ETensorType::UINT32 : {
         return "uint32_t";
      }
      case ETensorType::UINT64 : {
         return "uint64_t";
      }
      case ETensorType::DOUBLE : {
         return "double";
      }
      default:{
         return "other";
      }
   }
}

ETensorType ConvertStringToType(std::string type){
   if(type == "float32" || type == "float" || type == "Float"){
     return ETensorType::FLOAT;
   }
   else if(type == "int64"){
     return ETensorType::INT64;
   }
   else if (type == "double" || type == "float64"){
      return ETensorType::DOUBLE;
   }
   else{
      return ETensorType::UNDEFINED;
   }
}

std::string ConvertShapeToString(std::vector<size_t> shape) {
   std::stringstream out;
   out << "{ ";
   for (size_t i = 0; i < shape.size(); i++) {
      out << shape[i];
      if (i < shape.size()-1) out << " , ";
   }
   out << " }";
   return out.str();
}

namespace{
template<typename T>
static inline void copy_vector_data(int_t no_of_copies, int_t input_size, T* input, T* target){  //only visible within this translation unit
   std::memcpy(target, input, input_size * sizeof(T));
   int_t already_copied = 1;

   while (already_copied * 2 <= no_of_copies){
      std::memcpy(target + already_copied * input_size, target, already_copied * input_size * sizeof(T));
      already_copied *= 2;
   }

   if (already_copied < no_of_copies){
      std::memcpy(target + already_copied * input_size, target, (no_of_copies - already_copied) * input_size * sizeof(T));
   }
}
}

std::vector<size_t>  UTILITY::BidirectionalBroadcastShape(const std::vector<size_t>& shapeA, const std::vector<size_t>& shapeB)
{
   if (shapeA.size() == shapeB.size()) {
      if (ConvertShapeToLength(shapeA) != ConvertShapeToLength(shapeB)) {
         throw
            std::runtime_error("TMVA::SOFIE - Shape " + ConvertShapeToString(shapeA)
               + " and shape " + ConvertShapeToString(shapeB) + " are incompatible.");
      }
      return shapeA;
   }
   // The number of dimensions of A must be less or equal than the number of dimensions of B
   if (shapeA.size() > shapeB.size()) {
      throw
         std::runtime_error("TMVA::SOFIE - Error broadcasting tensor of shape "
            + ConvertShapeToString(shapeA) + " to " + ConvertShapeToString(shapeB));
   }
   std::vector<size_t> outShape(shapeB.size(), 1);

   // Find i and j such that A[k]=B[k] for k in [i, j) and A[k]=1 otherwise
   size_t i = 0;
   for (size_t k = 0; k < shapeB.size(); k++) {
      if (shapeB[k] == shapeA[0]) {
         outShape[k] = shapeB[k];
         i = k;
         break;
      }
   }
   for (size_t k = 1; k < shapeA.size(); k++) {
      if (shapeA[k] == shapeB[k + i]) {
         outShape[k + i] = shapeA[k];
      } else {
         break;
      }
   }

   if (ConvertShapeToLength(outShape) != ConvertShapeToLength(shapeA)) {
      std::stringstream ss;
      ss << "TMVA::SOFIE - Error broadcasting tensor of shape";
      ss << ConvertShapeToString(shapeA);
      ss << " to ";
      ss << ConvertShapeToString(shapeB);
      throw
         std::runtime_error(ss.str()); 
   }

   return outShape;
}

std::string UTILITY::Clean_name(std::string input_tensor_name){
   std::string s (input_tensor_name);
   s.erase(std::remove_if(s.begin(), s.end(), []( char const& c ) -> bool { return !std::isalnum(c); } ), s.end());
   return s;
}

std::vector<size_t> UTILITY::ComputeStrideFromShape(const std::vector<size_t> & shape) {
   // assume row major layout
   const auto size = shape.size();
   std::vector<size_t> strides(size,1);
   for (std::size_t i = 1; i < size; i++) {
      strides[size - 1 - i] = strides[size - 1 - i + 1] * shape[size - 1 - i + 1];
   }
   return strides;
}


}//SOFIE
}//Experimental
}//TMVA
