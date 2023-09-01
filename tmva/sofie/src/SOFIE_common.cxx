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
   // Empty shape represent scalar values, so we return a length=1
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

bool UTILITY::AreSameShape(const std::vector<size_t>& shapeA, const std::vector<size_t>& shapeB) {
   if (shapeA.size() != shapeB.size()) {
      return false;
   }
   for (size_t dim = 0; dim < shapeA.size(); dim++) {
      if (shapeA[dim] != shapeB[dim]) {
         return false;
      }
   }
   return true;
}

std::vector<size_t>  UTILITY::MultidirectionalBroadcastShape(std::vector<std::vector<size_t>> shape)
{
   if (shape.size() < 2) {
      throw
         std::runtime_error("TMVA::SOFIE - MultidirectionalBroadcastShape requires at least 2 input shapes.");
   }
   // Number of input shapes to broadcast
   size_t n = shape.size();
   // Size of the output shape
   size_t targetSize = shape[0].size();
   for (size_t i = 1; i < n; i++) {
      targetSize = std::max(targetSize, shape[i].size());
   }
   // Check if they have the same size
   bool sameSize = true;
   for (size_t i = 0; i < n; i++) {
      if (shape[i].size() != targetSize) {
         sameSize = false;
         break;
      }
   }
   if (sameSize) {
      // Check if they have the same shape
      bool sameShape = true;
      for (size_t i = 1; i < n; i++) {
         for (size_t dim = 0; dim < shape[0].size(); dim++) {
            if (shape[i][dim] != shape[0][dim]) {
               sameShape = false;
               break;
            }
         }
         if (!sameShape) {
            break;
         }
      }
      if (sameShape) {
         return shape[0];
      } else {
         // Set the target shape
         std::vector<size_t> targetShape(targetSize, 1);
         for (size_t i = 0; i < n; i++) {
            for (size_t dim = 0; dim < targetSize; dim++) {
               targetShape[dim] = std::max(targetShape[dim], shape[i][dim]);
            }
         }
         // Check if the input shapes are broadcastable to targetShape
         bool broadcastable = true;
         for (size_t i = 0; i < n; i++) {
            for (size_t dim = 0; dim < targetSize; dim++) {
               if (shape[i][dim] != 1 && targetShape[dim] != 1 && shape[i][dim] != targetShape[dim]) {
                  broadcastable = false;
                  break;
               }
               if (!broadcastable) {
                  break;
               }
            }
         }
         // They have the same shape and they are broadcastable to targetShape
         if (broadcastable) {
            return targetShape;
         } else {
            std::stringstream ss;
            ss << "TMVA::SOFIE - Error multidirectional broadcasting shapes ";
            for (size_t i = 0; i < n; i++) {
               ss << ConvertShapeToString(shape[i]);
               if (n > 2 && i < n - 2) {
                  ss << ", ";
               } else if ( n >=2 && i == n - 2) {
                  ss << " and ";
               }
            }
            ss << " to the same shape.";
            throw
               std::runtime_error(ss.str());
         }
      } // end sameShape
   } // end sameSize
   // Prepend the ith shape with ones
   for (size_t i = 0; i < n; i++) {
      if (shape[i].size() < targetSize) {
         std::vector<size_t> newShape(targetSize, 1);
         size_t offset = targetSize - shape[i].size();
         std::copy(shape[i].begin(), shape[i].end(), newShape.begin() + offset);
         shape[i] = newShape;
      }
   }
   // Set the target shape
   std::vector<size_t> targetShape(targetSize, 1);
   for (size_t i = 0; i < n; i++) {
      for (size_t dim = 0; dim < targetSize; dim++) {
         targetShape[dim] = std::max(targetShape[dim], shape[i][dim]);
      }
   }
   // Check if the shapes are broadcastable to targetShape
   bool broadcastable = true;
   for (size_t i = 0; i < n; i++) {
      for (size_t dim = 0; dim < targetSize; dim++) {
         if (shape[i][dim] != targetShape[dim] && shape[i][dim] != 1 && targetShape[dim] != 1) {
            broadcastable = false;
            break;
         }
      }
      if (!broadcastable) {
         break;
      }
   }
   if (broadcastable) {
      return targetShape;
   } else {
      std::stringstream ss;
      ss << "TMVA::SOFIE - Error multidirectional broadcasting shapes ";
      for (size_t i = 0; i < n; i++) {
         ss << ConvertShapeToString(shape[i]);
         if (n > 2 && i < n - 2) {
            ss << ", ";
         } else if ( n >=2 && i == n - 2) {
            ss << " and ";
         }
      }
      ss << " to the same shape.";
      throw
         std::runtime_error(ss.str());
   }
}

std::vector<size_t>  UTILITY::UnidirectionalBroadcastShape(std::vector<size_t> shapeA, std::vector<size_t> shapeB)
{
   size_t sizeA = shapeA.size();
   size_t sizeB = shapeB.size();
   // Check if A and B have the same shape
   if (UTILITY::AreSameShape(shapeA, shapeB)){
      return shapeA;
   }
   // Find the common shape of A and B
   size_t size = std::max(sizeA, sizeB);
   if (sizeA < size) {
      std::vector<size_t> newShapeA(size, 1);
      size_t offset = size - sizeA;
      std::copy(shapeA.begin(), shapeA.end(), newShapeA.begin() + offset);
      shapeA = std::move(newShapeA);
   }
   if (sizeB < size) {
      std::vector<size_t> newShapeB(size, 1);
      size_t offset = size - sizeB;
      std::copy(shapeB.begin(), shapeB.end(), newShapeB.begin() + offset);
      shapeB = std::move(newShapeB);
   }
   bool broadcastable = true;
   for (size_t i = 0; i < size; i++) {
      if (shapeA[i] != shapeB[i] && shapeA[i] != 1 && shapeB[i] != 1) {
         broadcastable = false;
         break;
      }
   }
   if (broadcastable) {
      // The output shape is max(outShape, targetShape)
      std::vector<size_t> targetShape(size, 1);
      for (size_t i = 0; i < size; i++) {
         targetShape[i] = std::max(shapeA[i], shapeB[i]);
      }
      return targetShape;
   } else {
      throw
         std::runtime_error("TMVA::SOFIE - Error unidirectional broadcasting tensors of shape "
            + ConvertShapeToString(shapeA) + " and " + ConvertShapeToString(shapeB)
            + " to a common shape.");
   }
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
