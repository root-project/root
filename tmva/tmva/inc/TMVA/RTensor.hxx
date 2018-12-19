#ifndef TMVA_RTENSOR
#define TMVA_RTENSOR

#include "ROOT/RVec.hxx"

#include <type_traits>
#include <cstdint>   // uint8_t
#include <cstddef>   // std::size_t
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream
#include <algorithm> // std::remove, std::equal

namespace TMVA {
namespace Experimental {

/// Memory order types
struct MemoryOrder {
   using type = uint8_t;
   const static type RowMajor = 0;
   const static type ColumnMajor = 1;
};

namespace Internal {

/// \brief Get size of tensor from shape vector
/// \param[in] shape Shape vector
/// \return Size of contiguous memory
template <typename T>
std::size_t GetSizeFromShape(const T &shape)
{
   if (shape.size() == 0)
      return 0;
   std::size_t size = 1;
   for (auto &s : shape)
      size *= s;
   return size;
}

/// \brief Compute strides from shape vector.
/// \param[in] shape Shape vector
/// \param[in] memoryOrder Memory order
/// \return Size of contiguous memory
///
/// This information is needed for the multi-dimensional indexing. See here:
/// https://en.wikipedia.org/wiki/Row-_and_column-major_order
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
template <typename T>
std::vector<std::size_t> ComputeStrides(const T &shape, MemoryOrder::type memoryOrder)
{
   const auto size = shape.size();
   T strides(size);
   if (memoryOrder == MemoryOrder::RowMajor) {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[size - 1 - i] = 1;
         } else {
            strides[size - 1 - i] = strides[size - 1 - i + 1] * shape[size - 1 - i + 1];
         }
      }
   } else if (memoryOrder == MemoryOrder::ColumnMajor) {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[i] = 1;
         } else {
            strides[i] = strides[i - 1] * shape[i - 1];
         }
      }
   } else {
      std::stringstream ss;
      ss << "Memory order type " << memoryOrder << " is not known for calculating the strides.";
      throw std::runtime_error(ss.str());
   }
   return strides;
}

/// \brief Type checking for all types of a parameter pack, e.g., used in combination with std::is_convertible
template <class... Ts>
struct and_types : std::true_type {
};
template <class T0, class... Ts>
struct and_types<T0, Ts...> : std::integral_constant<bool, T0{} && and_types<Ts...>{}> {
};

/// \brief Copy slice of a tensor recursively from here to there
/// \param[in] here Source tensor
/// \param[in] there Target tensor (slice of source tensor)
/// \param[in] mins Minimum of indices for each dimension
/// \param[in] maxs Maximum of indices for each dimension
/// \param[in] idx Current indices
/// \param[in] active Active index needed to stop the recursion
///
/// Copy the content of a slice of a tensor from source to target. This is done
/// by recursively iterating over the ranges of the slice for each dimension.
template <typename T>
void RecursiveCopy(T &here, T &there, std::vector<std::size_t> &mins, std::vector<std::size_t> &maxs,
                   std::vector<std::size_t> idx, std::size_t active)
{
   const auto size = idx.size();
   for (std::size_t i = mins[active]; i < maxs[active]; i++) {
      idx[active] = i;
      if (active == size - 1) {
         auto idxThere = idx;
         for (std::size_t j = 0; j < size; j++) {
            idxThere[j] -= mins[j];
         }
         there(idxThere) = here(idx);
      } else {
         Internal::RecursiveCopy(here, there, mins, maxs, idx, active + 1);
      }
   }
}

} // namespace TMVA::Experimental::Internal

/// \class TMVA::Experimental::RTensor
/// \brief RTensor is a container with contiguous memory and shape information.
/// \tparam T Data-type of the tensor
///
/// An RTensor is a vector-like container, which has additional shape information.
/// The elements of the multi-dimensional container can be accessed by their
/// indices in a coherent way without taking care about the one-dimensional memory
/// layout of the contiguous storage. This also allows to manipulate the shape
/// of the container without moving the actual elements in memory. Another feature
/// is that an RTensor can own the underlying contiguous memory but also represent
/// only a view on existing data without owning it.
template <typename T>
class RTensor {
public:
   // Typedefs
   using Container_t = typename ROOT::VecOps::RVec<T>;
   using Shape_t = typename std::vector<std::size_t>;
   using Index_t = Shape_t;
   using Slice_t = typename std::vector<int>;
   using iterator = typename Container_t::iterator;
   using const_iterator = typename Container_t::const_iterator;
   using reverse_iterator = typename Container_t::reverse_iterator;
   using const_reverse_iterator = typename Container_t::const_reverse_iterator;

   // Constructors
   RTensor(const Shape_t &shape, MemoryOrder::type memoryOrder = MemoryOrder::RowMajor);
   RTensor(T *data, const Shape_t &shape, MemoryOrder::type memoryOrder = MemoryOrder::RowMajor);

   // Access elements
   template <typename... Idx>
   T &At(Idx... idx);
   T &At(const Index_t &idx);
   template <typename... Idx>
   T &operator()(Idx... idx);
   T &operator()(const Index_t &idx);

   // Iterator interface
   iterator begin() noexcept { return fData.begin(); }
   const_iterator begin() const noexcept { return fData.begin(); }
   const_iterator cbegin() const noexcept { return fData.cbegin(); }
   iterator end() noexcept { return fData.end(); }
   const_iterator end() const noexcept { return fData.end(); }
   const_iterator cend() const noexcept { return fData.cend(); }
   reverse_iterator rbegin() noexcept { return fData.rbegin(); }
   const_reverse_iterator rbegin() const noexcept { return fData.rbegin(); }
   const_reverse_iterator crbegin() const noexcept { return fData.crbegin(); }
   reverse_iterator rend() noexcept { return fData.rend(); }
   const_reverse_iterator rend() const noexcept { return fData.rend(); }
   const_reverse_iterator crend() const noexcept { return fData.crend(); }

   // Slicing
   RTensor<T> Slice(const Slice_t &slice);
   template <typename... Idx>
   RTensor<T> Slice(Idx... slice);

   // Shape modifications
   void Reshape(const Shape_t &shape);
   void ExpandDims(int axis);
   void Squeeze();
   void Transpose();

   // Get properties of container
   std::size_t GetSize() const { return fData.size(); }        // Return size of contiguous memory
   Shape_t GetShape() const { return fShape; }                 // Return shape
   Shape_t GetStrides() const { return fStrides; }             // Return strides
   T *GetData() { return fData.data(); }                       // Return pointer to data
   Container_t &GetDataAsVec() { return fData; }               // Return data as reference to vector
   MemoryOrder::type GetMemoryOrder() { return fMemoryOrder; } // Return memory order

private:
   Shape_t fShape;                 // Shape of tensor
   Shape_t fStrides;               // Strides, needed for indexing
   Container_t fData;              // Container for contiguous memory
   MemoryOrder::type fMemoryOrder; // Memory ordering
};

/// \brief Construct a tensor from given shape initialized with zeros
/// \param[in] shape Shape vector
/// \param[in] memoryOrder Memory order
template <typename T>
RTensor<T>::RTensor(const Shape_t &shape, MemoryOrder::type memoryOrder)
{
   fShape = shape;
   fStrides = Internal::ComputeStrides(shape, memoryOrder);
   const auto size = Internal::GetSizeFromShape(shape);
   fData = Container_t(size);
   fMemoryOrder = memoryOrder;
}

/// \brief Construct a tensor adopting given data
/// \param[in] data Pointer to data contiguous in memory
/// \param[in] shape Shape vector
/// \param[in] memoryOrder Memory order
template <typename T>
RTensor<T>::RTensor(T *data, const Shape_t &shape, MemoryOrder::type memoryOrder)
{
   fShape = shape;
   fStrides = Internal::ComputeStrides(shape, memoryOrder);
   const auto size = Internal::GetSizeFromShape(shape);
   fData = Container_t(data, size);
   fMemoryOrder = memoryOrder;
}

/// \brief Reshape tensor
/// \param[in] shape Shape vector
template <typename T>
void RTensor<T>::Reshape(const Shape_t &shape)
{
   const auto sizeInput = Internal::GetSizeFromShape(shape);
   if (sizeInput != GetSize()) {
      std::stringstream ss;
      ss << "Cannot reshape tensor with size " << GetSize() << " into shape { ";
      for (std::size_t i = 0; i < shape.size(); i++) {
         if (i != shape.size() - 1) {
            ss << shape[i] << ", ";
         } else {
            ss << shape[i] << " }.";
         }
      }
      throw std::runtime_error(ss.str());
   }
   fShape = shape;
   fStrides = Internal::ComputeStrides(shape, fMemoryOrder);
}

/// \brief Expand dimensions
/// \param[in] idx Index in shape vector where dimension is added
template <typename T>
void RTensor<T>::ExpandDims(int idx)
{
   // TODO: What happens if shape has size 0?
   const int len = fShape.size();
   if (idx < 0) {
      if (len - idx < 0)
         throw std::runtime_error("Given negative index is invalid.");
      fShape.insert(fShape.end() + 1 + idx, 1);
   } else {
      if (idx > len - 1)
         throw std::runtime_error("Given index is invalid.");
      fShape.insert(fShape.begin() + idx, 1);
   }

   // Recompute strides
   fStrides = Internal::ComputeStrides(fShape, fMemoryOrder);
}

/// \brief Squeeze dimensions
template <typename T>
void RTensor<T>::Squeeze()
{
   // If shape is empty, return early
   if (fShape.size() == 0)
      return;

   // Remove dimensions of 1
   fShape.erase(std::remove(fShape.begin(), fShape.end(), 1), fShape.end());

   // If all dimensions are 1, we need to keep one.
   if (fShape.size() == 0)
      fShape.emplace_back(1);

   // Recompute strides
   fStrides = Internal::ComputeStrides(fShape, fMemoryOrder);
}

/// \brief Transpose
template <typename T>
void RTensor<T>::Transpose()
{
   // Invert memory order
   if (fMemoryOrder == MemoryOrder::RowMajor) {
      fMemoryOrder = MemoryOrder::ColumnMajor;
   } else if (fMemoryOrder == MemoryOrder::ColumnMajor) {
      fMemoryOrder = MemoryOrder::RowMajor;
   } else {
      std::runtime_error("Column order is not known.");
   }

   // Reverse shape
   std::reverse(fShape.begin(), fShape.end());

   // Recompute strides
   fStrides = Internal::ComputeStrides(fShape, fMemoryOrder);
}

/// \brief Access elements
/// \param[in] idx Index vector
/// \return Reference to element
template <typename T>
T &RTensor<T>::At(const Index_t &idx)
{
   std::size_t globalIndex = 0;
   const auto size = idx.size();
   for (std::size_t i = 0; i < size; i++) {
      globalIndex += fStrides[size - 1 - i] * idx[size - 1 - i];
   }
   return fData[globalIndex];
}

/// \brief Access elements
/// \param[in] idx Indices
/// \return Reference to element
template <typename T>
template <typename... Idx>
T &RTensor<T>::At(Idx... idx)
{
   static_assert(Internal::and_types<std::is_convertible<Idx, std::size_t>...>{},
                 "Given index is not convertible to std::size_t.");
   return this->At({static_cast<std::size_t>(idx)...});
}

/// \brief Access elements
/// \param[in] idx Index vector
/// \return Reference to element
template <typename T>
T &RTensor<T>::operator()(const Index_t &idx)
{
   return this->At(idx);
}

/// \brief Access elements
/// \param[in] idx Indices
/// \return Reference to element
template <typename T>
template <typename... Idx>
T &RTensor<T>::operator()(Idx... idx)
{
   static_assert(Internal::and_types<std::is_convertible<Idx, std::size_t>...>{},
                 "Given index is not convertible to std::size_t.");
   return this->At({static_cast<std::size_t>(idx)...});
}

/// \brief Slicing
/// \param[in] idx Index vector
/// \return New RTensor with elements of slice
template <typename T>
RTensor<T> RTensor<T>::Slice(const Slice_t &idx)
{
   const auto size = idx.size();
   if (size != fShape.size()) {
      throw std::runtime_error("Rank of given indices does not match shape.");
   }

   for (std::size_t i = 0; i < size; i++) {
      if (idx[i] < 0 && idx[i] != -1) {
         throw std::runtime_error("Negative indices of a slice can only be -1.");
      }
   }

   // Calculate new shape of slice
   Shape_t newShape;
   for (std::size_t i = 0; i < size; i++) {
      if (idx[i] == -1) {
         newShape.emplace_back(fShape[i]);
      } else {
         newShape.emplace_back(1);
      }
   }

   // Allocate new tensor
   RTensor<T> x(newShape, fMemoryOrder);

   // Copy over values
   Shape_t mins(size), maxs(size);
   for (std::size_t i = 0; i < size; i++) {
      if (idx[i] == -1) {
         mins[i] = 0;
         maxs[i] = fShape[i];
      } else {
         mins[i] = idx[i];
         maxs[i] = idx[i] + 1;
      }
   }
   Internal::RecursiveCopy(*this, x, mins, maxs, mins, 0);

   // Remove dimensions of 1
   x.Squeeze();

   return x;
}

/// \brief Slicing
/// \param[in] idx Indices
/// \return New RTensor with elements of slice
template <typename T>
template <typename... Idx>
RTensor<T> RTensor<T>::Slice(Idx... idx)
{
   static_assert(Internal::and_types<std::is_convertible<Idx, int>...>{}, "Given index is not convertible to int.");
   return this->Slice({static_cast<int>(idx)...});
}

/// \brief Pretty printing
/// \param[in] os Output stream
/// \param[in] x RTensor
/// \return Modified output stream
template <typename T>
std::ostream &operator<<(std::ostream &os, RTensor<T> &x)
{
   const auto shapeSize = x.GetShape().size();
   if (shapeSize == 1) {
      os << "{ ";
      auto data = x.GetData();
      const auto size = x.GetSize();
      for (std::size_t i = 0; i < size; i++) {
         os << *(data + i);
         if (i != size - 1)
            os << ", ";
      }
      os << " }";
   } else if (shapeSize == 2) {
      os << "{";
      const auto shape = x.GetShape();
      for (std::size_t i = 0; i < shape[0]; i++) {
         os << " { ";
         for (std::size_t j = 0; j < shape[1]; j++) {
            os << x(i, j);
            if (j < shape[1] - 1) {
               os << ", ";
            } else {
               os << " ";
            }
         }
         os << "}";
      }
      os << " }";
   } else {
      os << "{ printing not yet implemented for this rank }";
   }
   return os;
}

} // namespace TMVA::Experimental
} // namespace TMVA

namespace cling {
template <typename T>
std::string printValue(TMVA::Experimental::RTensor<T> *x)
{
   std::stringstream ss;
   ss << *x;
   return ss.str();
}
} // namespace cling

#endif // TMVA_RTENSOR
