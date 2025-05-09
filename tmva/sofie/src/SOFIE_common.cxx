#include "TMVA/SOFIE_common.hxx"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <charconv>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

/// @brief  Convert shape from integer format to dynamic one (based on Dim)
/// @param shape
/// @return shape based on Dim
std::vector<Dim> ConvertShapeToDim(const std::vector<size_t> & shape){
   std::vector<Dim> ret_shape(shape.size());
   for (size_t i =0; i < shape.size(); i++){
      ret_shape[i].dim = shape[i];
   }
   return ret_shape;
}

/// @brief Convert shape based on Dim to integer format
/// @param shape
/// @return shape based on integer. Return an empty shape in case shape is dynamic (has a parameter)
std::vector<size_t> ConvertShapeToInt(const std::vector<Dim> & shape){
   std::vector<size_t> ret_shape(shape.size());
   for (size_t i =0; i < shape.size(); i++){
      if (shape[i].isParam) {
         // try converting to integer in case string is a number >=0
         int val = -1;
         try {
            val = std::stoi(shape[i].param);
            if (val >= 0) ret_shape[i] = static_cast<size_t>(val);
            else {
               ret_shape.clear();
               break;
            }
         }
         catch (const std::invalid_argument& ) {
            ret_shape.clear();
            break;
         }
      } else {
         ret_shape[i] = shape[i].dim;
      }
   }
   return ret_shape;
}


std::size_t ConvertShapeToLength(const std::vector<size_t> & shape){
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
      case ETensorType::INT8 : {
         return "int8_t";
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
      case ETensorType::UINT8 : {
         return "uint8_t";
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
      case ETensorType::BOOL : {
         return "bool";
      }
      default:{
         return "other_" + std::to_string( (int) type);
      }
   }
}

ETensorType ConvertStringToType(std::string type){
   if(type == "float32" || type == "float" || type == "Float"){
     return ETensorType::FLOAT;
   }
   else if(type == "int64" || type == "int64_t"){
     return ETensorType::INT64;
   }
   else if (type == "double" || type == "float64"){
      return ETensorType::DOUBLE;
   }
   else if (type == "bool" ){
      return ETensorType::BOOL;
   }
   else{
      return ETensorType::UNDEFINED;
   }
}

std::string ConvertShapeToString(const std::vector<size_t> & shape) {
   std::stringstream out;
   out << "{ ";
   for (size_t i = 0; i < shape.size(); i++) {
      out << shape[i];
      if (i < shape.size()-1) out << " , ";
   }
   out << " }";
   return out.str();
}

std::string ConvertDimShapeToString(const std::vector<Dim> & shape) {
   std::stringstream out;
   out << "{ ";
   for (size_t i = 0; i < shape.size(); i++) {
      out << shape[i].GetVal();
      if (i < shape.size()-1) out << " , ";
   }
   out << " }";
   return out.str();
}

std::string ConvertDimShapeToLength(const std::vector<Dim> & shape) {
   // convert generic shape to a string
   // multiply all the integer specified dimensions of the shape
   std::string length;
   // case of empty vectors return 1
   if (shape.empty()) return "1";
   size_t int_length = 0;
   for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i].isParam) {
         if (!length.empty()) length += " * ";
         length += shape[i].param;
      } else {
         if (int_length == 0)
            int_length = shape[i].dim;
         else
            int_length *= shape[i].dim;
      }
   }
   // multiply the integer components to the parametric one
   // if larger than 1
   if (int_length > 0) {
      if (!length.empty() && int_length > 1) {
         length += " * ";
         length += std::to_string(int_length);
      } else if (length.empty()) { // case is full known shape
         length = std::to_string(int_length);
      }
   }
   return length;
}
std::string ConvertShapeToString(const std::vector<Dim> & shape) {
   return ConvertDimShapeToString(shape);
}
std::string ConvertDynamicShapeToLength(const std::vector<Dim> & shape) {
   return ConvertDimShapeToLength(shape);
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

bool IsInteger(const std::string & s) {
   int value;
   auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
   return ec == std::errc() && ptr == s.data() + s.size();
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
bool UTILITY::AreSameShape(const std::vector<size_t>& shapeA, const std::vector<Dim>& shapeB) {
   if (shapeA.size() != shapeB.size()) {
      return false;
   }
   for (size_t dim = 0; dim < shapeA.size(); dim++) {
      if (shapeB[dim].isParam) return false;
      if (shapeA[dim] != shapeB[dim].dim) {
         return false;
      }
   }
   return true;
}
bool UTILITY::AreSameShape(const std::vector<Dim>& shapeA, const std::vector<Dim>& shapeB) {
   if (shapeA.size() != shapeB.size()) {
      return false;
   }
   for (size_t dim = 0; dim < shapeA.size(); dim++) {
      if (shapeA[dim].GetVal() != shapeB[dim].GetVal()) {
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

// check multi-directional broadcasting of two shapes (need to pass inputs by non const ref. since we might prepends with one's
// return a pair of integer flag and new broadcasted shape
// if flag = 0: shape are identical
//    flag = 1: return shape is equal to A, we broadcast B
//    flag = 2: return shape is equal to B we broadcast A
//    flag = 3: return shape is common of two we broadcast A and B to output
std::pair<int, std::vector<size_t>>  UTILITY::MultidirectionalBroadcastShape(std::vector<size_t> & shapeA, std::vector<size_t> & shapeB)
{
   size_t sizeA = shapeA.size();
   size_t sizeB = shapeB.size();
   // Check if A and B have the same shape
   if (UTILITY::AreSameShape(shapeA, shapeB)){
      return std::make_pair(0, shapeA);
   }
   // Find the common shape of A and B
   size_t size = std::max(sizeA, sizeB);
   if (sizeA < size) {
      // prepend 1's in A to make of same shape as B
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
   int broadcastFlag = 0;
   if (broadcastable) {
      // The output shape is max(outShape, targetShape)
      std::vector<size_t> targetShape(size, 1);
      for (size_t i = 0; i < size; i++) {
         targetShape[i] = std::max(shapeA[i], shapeB[i]);
         if (shapeB[i] < targetShape[i]) broadcastFlag |= 1;
         if (shapeA[i] < targetShape[i]) broadcastFlag |= 2;
      }
      return std::make_pair(broadcastFlag, targetShape);
   } else {
      throw
         std::runtime_error("TMVA::SOFIE - Error multidirectional broadcasting tensors of shape "
            + ConvertShapeToString(shapeA) + " and " + ConvertShapeToString(shapeB)
            + " to a common shape.");
   }
}
// unidirectional broadcast- only B changes
std::vector<size_t>  UTILITY::UnidirectionalBroadcastShape(std::vector<size_t> & shapeA, std::vector<size_t> & shapeB)
{
   auto ret = UTILITY::MultidirectionalBroadcastShape(shapeA, shapeB);
   if (ret.first > 1) {
      std::runtime_error("TMVA::SOFIE - Error unidirectional broadcasting tensors of shape "
            + ConvertShapeToString(shapeA) + " and " + ConvertShapeToString(shapeB)
            + " to a common shape.");
   }
   return ret.second;
}

// for broadcasting Dim shapes
// flag indicates also which vector needs to be broadcasted
//    flag & 1 == 1 : broadcast B -> A
//    flag & 2 == 2 : broadcast A -> B
//    flag & 4 == 4 a run time check is needed on shapes with values
std::pair<int, std::vector<Dim>> UTILITY::MultidirectionalBroadcastShape(std::vector<Dim> & shapeA, std::vector<Dim> & shapeB) {
   size_t sizeA = shapeA.size();
   size_t sizeB = shapeB.size();
   // Check if A and B have the same shape
   if (UTILITY::AreSameShape(shapeA, shapeB)){
      return std::make_pair(0, shapeA);
   }
   // Find the common shape of A and B
   size_t size = std::max(sizeA, sizeB);
   if (sizeA < size) {
      // prepend 1's in A to make of same shape as B
      std::vector<Dim> newShapeA(size, Dim{1});
      size_t offset = size - sizeA;
      std::copy(shapeA.begin(), shapeA.end(), newShapeA.begin() + offset);
      shapeA = std::move(newShapeA);
   }
   if (sizeB < size) {
      std::vector<Dim> newShapeB(size, Dim{1});
      size_t offset = size - sizeB;
      std::copy(shapeB.begin(), shapeB.end(), newShapeB.begin() + offset);
      shapeB = std::move(newShapeB);
   }

   int broadcastFlag = 0;
   // The output shape is targetShape
   std::vector<Dim> targetShape(size);
   for (size_t i = 0; i < size; i++) {
      // assume we broadcast to the parametric value
      if (shapeA[i] == shapeB[i]) {
         targetShape[i] = shapeA[i];
      } else if (shapeA[i].isParam && shapeB[i].GetVal() == "1" ) {
         // broadcast B to A (case A is parametric with )
         targetShape[i] = shapeA[i];
         broadcastFlag |= 1;
      } else if (shapeA[i].GetVal() == "1" && shapeB[i].isParam) {
         // broadcast A to B
         targetShape[i] = shapeB[i];
         broadcastFlag |= 2;
      } else if (!shapeA[i].isParam && !shapeB[i].isParam) {
         if (shapeB[i].dim == 1) {
            targetShape[i] = shapeA[i];
            broadcastFlag |= 1;
         } else if (shapeA[i].dim == 1) {
            targetShape[i] = shapeB[i];
            broadcastFlag |= 2;
         } else {
            // non broadcastable case cannot have A and B two different defined shapes different than one
            broadcastFlag = -1;
         }
      } else if (shapeA[i].isParam && shapeB[i].isParam) {
         // full dynamic case - we will decided at run time
         std::stringstream s;
         s <<  "std::max(" << shapeA[i] << "," << shapeB[i] << ")";
         // use -1 for dim to indicate is an expression
         targetShape[i] = Dim { s.str() , static_cast<size_t>(-1)};
         broadcastFlag |= 4;
      } else if (shapeA[i].isParam && !shapeB[i].isParam) {
         // A -> B need to check at run time if consistent
         targetShape[i] = shapeB[i];
         broadcastFlag |= 6;
      } else if (!shapeA[i].isParam && shapeB[i].isParam) {
         // B -> A need to check at run time if consistent
         targetShape[i] = shapeA[i];
         broadcastFlag |= 5;
      } else {
         // all cases should be covered
         throw std::runtime_error("TMVA::SOFIE - Fatal error in MultiDirectionalBroadCastDimShape");
      }
   }
   if (broadcastFlag == -1) {
      throw std::runtime_error("TMVA::SOFIE - Error multidirectional broadcasting tensors of shape " +
                                 ConvertDimShapeToString(shapeA) + " and " + ConvertDimShapeToString(shapeB) +
                                 " to a common shape.");
   }

   return std::make_pair(broadcastFlag, targetShape);
}

std::string UTILITY::Clean_name(std::string input_tensor_name){
   std::string s (input_tensor_name);
   std::replace( s.begin(), s.end(), '-', '_');
   // replace all non-alpohanumeric character except for "_"
   s.erase(std::remove_if(s.begin(), s.end(), []( char const& c ) -> bool { return !std::isalnum(c) && c != '_'; } ), s.end());
   return s;
}

std::vector<size_t> UTILITY::ComputeStrideFromShape(const std::vector<size_t> & shape) {
   // assume row major layout
   const auto size = shape.size();
   std::vector<size_t> strides(size,1);
   for (std::size_t i = 1; i < size; i++) {
      strides[size - 1 - i] = strides[size - i ] * shape[size - i];
   }
   return strides;
}

std::vector<Dim> UTILITY::ComputeStrideFromShape(const std::vector<Dim> & shape) {
   // assume row major layout
   const auto size = shape.size();
   std::vector<Dim> strides(size);
   if (size > 0) {
      strides[size-1] = Dim{1};
      for (std::size_t i = 1; i < size; i++) {
         if (!shape[size-i].isParam && !strides[size-i].isParam)
            strides[size - 1 - i] = Dim{strides[size-i].dim * shape[size-i].dim};
         else {
            if (strides[size-i].GetVal() == "1")
               strides[size - 1 - i] = shape[size-i];
            else if (shape[size-i].GetVal() == "1")
               strides[size - 1 - i] = strides[size-i];
            else
              strides[size - 1 - i] = Dim{std::string(strides[size-i].GetVal() + "*" + shape[size-i].GetVal())};
         }
      }
   }
   return strides;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
