#include "TMVA/SOFIE_common.hxx"
#include<cctype>
#include <sstream>

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
      default:{
         return "other";
      }
   }
}

ETensorType ConvertStringToType(std::string type){
   if(type == "float32" || type == "Float"){
     return ETensorType::FLOAT;
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

template <typename T>
T* UTILITY::Unidirectional_broadcast(const T* original_data, const std::vector<size_t> original_shape, const std::vector<size_t> target_shape)
{

      int original_length = 1;
      int target_length = 1;
      for (size_t i = 0; i < original_shape.size(); i++){
         original_length *= original_shape[i];
      }
      for (size_t i = 0; i < target_shape.size(); i++){
         target_length *= target_shape[i];
      }
      if (original_shape.size() > target_shape.size())  {
         std::string  targetShape = "target : " + ConvertShapeToString(target_shape);
         std::string  originalShape = "original : " + ConvertShapeToString(original_shape);
         throw std::runtime_error(
            "TMVA::SOFIE Error in Broadcasting Tensor : original array has more dimensions than target shape," + originalShape + ", " + targetShape);
      }
      // if shape's sizes are different prepend 1 to get tensor with same shape size
      // since the broadcast is unidirectional we can only prepend
      std::vector<size_t> current_shape(original_shape);
      auto it = current_shape.begin();
      while (current_shape.size() < target_shape.size()) {
         it = current_shape.insert(it, 1);
      }
      // this code below will work
      // when shape are not equal e.g. (3,4,5,6) and (3) and we add 1 in all missing positions
      // since broadcasting is uni-directional we do not use it
      // std::vector<size_t> current_shape(target_shape.size(),1);
      // for (size_t i = 0; i < original_shape.size(); i++) {
      //    for (size_t j = 0; j < target_shape.size(); j++) {
      //       if (target_shape[j] == original_shape[i])
      //          current_shape[j] = original_shape[i];
      //    }
      // }

      T* new_datavector = new T[target_length];
      std::memcpy(new_datavector, original_data, original_length * sizeof(T));

      for (int dim = (int) target_shape.size() - 1; dim >= 0; dim--){
         if (current_shape[dim] != target_shape[dim]){
            if (current_shape[dim] != 1) {
               std::string targetShape = "target : " + ConvertShapeToString(target_shape);
               std::string originalShape = "original : " + ConvertShapeToString(current_shape);
               throw std::runtime_error ("TMVA::SOFIE Error in Broadcasting Tensor at least one dimension to be broadcast of the original array is not 1, " + originalShape + ", " + targetShape);
            }
            int_t group_size = 1;
            int_t no_of_groups = 1;
            int_t no_of_copies = target_shape[dim];

            for (size_t i = dim + 1; i < target_shape.size(); i++){
               group_size *= current_shape[i];
            }
            for (int i = 0; i < dim; i++){
               no_of_groups *= current_shape[i];
            }

            for (int curr_group = no_of_groups - 1; curr_group >= 0; curr_group--){
               copy_vector_data<T>(no_of_copies, group_size, new_datavector + curr_group * group_size,new_datavector + curr_group * group_size * no_of_copies);
            }

            current_shape[dim] = target_shape[dim];
         }
      }
      return new_datavector;
}

std::string UTILITY::Clean_name(std::string input_tensor_name){
   std::string s (input_tensor_name);
   s.erase(std::remove_if(s.begin(), s.end(), []( char const& c ) -> bool { return !std::isalnum(c); } ), s.end());
   return s;
}

template float* UTILITY::Unidirectional_broadcast(const float* original_data, const std::vector<size_t> original_shape, const std::vector<size_t> target_shape);

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
