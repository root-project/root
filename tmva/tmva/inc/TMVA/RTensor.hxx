#ifndef TMVA_RTENSOR
#define TMVA_RTENSOR

#include <vector>
#include <cstddef>     // std::size_t
#include <stdexcept>   // std::runtime_error
#include <sstream>     // std::stringstream
#include <memory>      // std::shared_ptr
#include <type_traits> // std::is_convertible
#include <algorithm>   // std::reverse

namespace TMVA {
namespace Experimental {

/// Memory layout type
enum class MemoryLayout : uint8_t {
   RowMajor = 0x01,
   ColumnMajor = 0x02
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
/// \param[in] layout Memory layout
/// \return Size of contiguous memory
///
/// This information is needed for the multi-dimensional indexing. See here:
/// https://en.wikipedia.org/wiki/Row-_and_column-major_order
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
template <typename T>
std::vector<std::size_t> ComputeStridesFromShape(const T &shape, MemoryLayout layout)
{
   const auto size = shape.size();
   T strides(size);
   if (layout == MemoryLayout::RowMajor) {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[size - 1 - i] = 1;
         } else {
            strides[size - 1 - i] = strides[size - 1 - i + 1] * shape[size - 1 - i + 1];
         }
      }
   } else if (layout == MemoryLayout::ColumnMajor) {
      for (std::size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[i] = 1;
         } else {
            strides[i] = strides[i - 1] * shape[i - 1];
         }
      }
   } else {
      std::stringstream ss;
      ss << "Memory layout type is not valid for calculating strides.";
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
/// is that an RTensor can own the underlying contiguous memory but can also represent
/// only a view on existing data without owning it.
template <typename V, typename C = std::vector<V>>
class RTensor {
public:
   // Typedefs
   using Value_t = V;
   using Shape_t = std::vector<std::size_t>;
   using Index_t = Shape_t;
   using Slice_t = std::vector<Shape_t>;
   using Container_t = C;

private:
   Shape_t fShape;
   Shape_t fStrides;
   std::size_t fSize;
   MemoryLayout fLayout;
   Value_t *fData;
   std::shared_ptr<Container_t> fContainer;

public:
   // Constructors

   /// \brief Construct a tensor as view on data
   /// \param[in] data Pointer to data contiguous in memory
   /// \param[in] shape Shape vector
   /// \param[in] layout Memory layout
   RTensor(Value_t *data, Shape_t shape, MemoryLayout layout = MemoryLayout::RowMajor)
      : fShape(shape), fLayout(layout), fData(data), fContainer(NULL)
   {
      fSize = Internal::GetSizeFromShape(shape);
      fStrides = Internal::ComputeStridesFromShape(shape, layout);
   }

   /// \brief Construct a tensor owning externally provided data
   /// \param[in] container Shared pointer to data container
   /// \param[in] shape Shape vector
   /// \param[in] layout Memory layout
   RTensor(std::shared_ptr<Container_t> container, Shape_t shape,
           MemoryLayout layout = MemoryLayout::RowMajor)
      : fShape(shape), fLayout(layout), fContainer(container)
   {
      fSize = Internal::GetSizeFromShape(shape);
      fStrides = Internal::ComputeStridesFromShape(shape, layout);
      fData = &(*container->begin());
   }

   /// \brief Construct a tensor owning data initialized with new container
   /// \param[in] shape Shape vector
   /// \param[in] layout Memory layout
   RTensor(Shape_t shape, MemoryLayout layout = MemoryLayout::RowMajor)
      : fShape(shape), fLayout(layout)
   {
      // TODO: Document how data pointer is determined using STL iterator interface.
      // TODO: Sanitize given container type with type traits
      fSize = Internal::GetSizeFromShape(shape);
      fStrides = Internal::ComputeStridesFromShape(shape, layout);
      fContainer = std::make_shared<Container_t>(fSize);
      fData = &(*fContainer->begin());
   }

   // Access elements
   Value_t &operator()(const Index_t &idx);
   template <typename... Idx> Value_t &operator()(Idx... idx);

   // Access properties
   std::size_t GetSize() { return fSize; }
   Shape_t GetShape() { return fShape; }
   Shape_t GetStrides() { return fStrides; }
   Value_t *GetData() { return fData; }
   std::shared_ptr<Container_t> GetContainer() { return fContainer; }
   MemoryLayout GetMemoryLayout() { return fLayout; }
   bool IsView() { return fContainer == NULL; }
   bool IsOwner() { return !IsView(); }

   // Transformations
   RTensor<Value_t, Container_t> Transpose();
   RTensor<Value_t, Container_t> Squeeze();
   RTensor<Value_t, Container_t> ExpandDims(int idx);
   RTensor<Value_t, Container_t> Reshape(const Shape_t &shape);
   RTensor<Value_t, Container_t> Slice(const Slice_t &slice);
};

/// \brief Access elements
/// \param[in] idx Index vector
/// \return Reference to element
template <typename Value_t, typename Container_t>
inline Value_t &RTensor<Value_t, Container_t>::operator()(const Index_t &idx)
{
   std::size_t globalIndex = 0;
   const auto size = idx.size();
   for (std::size_t i = 0; i < size; i++) {
      globalIndex += fStrides[size - 1 - i] * idx[size - 1 - i];
   }
   return *(fData + globalIndex);
}

/// \brief Access elements
/// \param[in] idx Indices
/// \return Reference to element
template <typename Value_t, typename Container_t>
template <typename... Idx>
Value_t &RTensor<Value_t, Container_t>::operator()(Idx... idx)
{
   static_assert(Internal::and_types<std::is_convertible<Idx, std::size_t>...>{},
                 "Indices are not convertible to std::size_t.");
   return this->operator()({static_cast<std::size_t>(idx)...});
}

/// \brief Transpose
/// \returns New RTensor
/// The tensor is transposed by inverting the associated memory layout from row-
/// major to column-major and vice versa. Therefore, the underlying data is not
/// touched.
template <typename Value_t, typename Container_t>
inline RTensor<Value_t, Container_t> RTensor<Value_t, Container_t>::Transpose()
{
   // Transpose by inverting memory layout
   if (fLayout == MemoryLayout::RowMajor) {
      fLayout = MemoryLayout::ColumnMajor;
   } else if (fLayout == MemoryLayout::ColumnMajor) {
      fLayout = MemoryLayout::RowMajor;
   } else {
      std::runtime_error("Memory layout is not known.");
   }

   // Create copy of container
   RTensor<Value_t, Container_t> x(*this);

   // Reverse shape
   std::reverse(x.fShape.begin(), x.fShape.end());

   // Reverse strides
   std::reverse(x.fStrides.begin(), x.fStrides.end());

   return x;
}

/// \brief Squeeze dimensions
/// \returns New RTensor
/// Squeeze removes the dimensions of size one from the shape.
template <typename Value_t, typename Container_t>
inline RTensor<Value_t, Container_t> RTensor<Value_t, Container_t>::Squeeze()
{
   // Remove dimensions of one and associated strides
   Shape_t shape;
   Shape_t strides;
   for (std::size_t i = 0; i < fShape.size(); i++) {
      if (fShape[i] != 1) {
         shape.emplace_back(fShape[i]);
         strides.emplace_back(fStrides[i]);
      }
   }

   // If all dimensions are 1, we need to keep one.
   // This does not apply if the inital shape is already empty. Then, return
   // the empty shape.
   if (shape.size() == 0 && fShape.size() != 0) {
      shape.emplace_back(1);
      strides.emplace_back(1);
   }

   // Create copy, attach new shape and strides and return
   RTensor<Value_t, Container_t> x(*this);
   x.fShape = shape;
   x.fStrides = strides;
   return x;
}

/// \brief Expand dimensions
/// \param[in] idx Index in shape vector where dimension is added
/// \returns New RTensor
/// Inserts a dimension of one into the shape.
template <typename Value_t, typename Container_t>
inline RTensor<Value_t, Container_t> RTensor<Value_t, Container_t>::ExpandDims(int idx)
{
   // Compose shape vector with additional dimensions and adjust strides
   const int len = fShape.size();
   auto shape = fShape;
   auto strides = fStrides;
   if (idx < 0) {
      if (len + idx + 1 < 0) {
         throw std::runtime_error("Given negative index is invalid.");
      }
      shape.insert(shape.end() + 1 + idx, 1);
      strides.insert(strides.begin() + 1 + idx, 1);
   } else {
      if (idx > len) {
         throw std::runtime_error("Given index is invalid.");
      }
      shape.insert(shape.begin() + idx, 1);
      strides.insert(strides.begin() + idx, 1);
   }

   // Create copy, attach new shape and strides and return
   RTensor<Value_t, Container_t> x(*this);
   x.fShape = shape;
   x.fStrides = strides;
   return x;
}

/// \brief Reshape tensor
/// \param[in] shape Shape vector
/// \returns New RTensor
/// Reshape tensor without changing the overall size
template <typename Value_t, typename Container_t>
inline RTensor<Value_t, Container_t> RTensor<Value_t, Container_t>::Reshape(const Shape_t &shape)
{
   const auto size = Internal::GetSizeFromShape(shape);
   if (size != fSize) {
      std::stringstream ss;
      ss << "Cannot reshape tensor with size " << fSize << " into shape { ";
      for (std::size_t i = 0; i < shape.size(); i++) {
         if (i != shape.size() - 1) {
            ss << shape[i] << ", ";
         } else {
            ss << shape[i] << " }.";
         }
      }
      throw std::runtime_error(ss.str());
   }

   // Compute new strides from shape
   auto strides = Internal::ComputeStridesFromShape(shape, fLayout);

   // Create copy, attach new shape and strides and return
   RTensor<Value_t, Container_t> x(*this);
   x.fShape = shape;
   x.fStrides = strides;
   return x;
}

/// \brief Create a slice of the tensor
/// \param[in] slice Slice vector
/// \returns New RTensor
/// A slice is a subset of the tensor defined by a vector of pairs of indices.
template <typename Value_t, typename Container_t>
inline RTensor<Value_t, Container_t> RTensor<Value_t, Container_t>::Slice(const Slice_t &slice)
{
   // Sanitize size of slice
   const auto sliceSize = slice.size();
   const auto shapeSize = fShape.size();
   if (sliceSize != shapeSize) {
      std::stringstream ss;
      ss << "Size of slice (" << sliceSize << ") is unequal number of dimensions (" << shapeSize << ").";
      throw std::runtime_error(ss.str());
   }

   // Sanitize slice indices
   // TODO: Sanitize slice indices
   /*
   for (std::size_t i = 0; i < sliceSize; i++) {
   }
   */

   // Convert -1 in slice to proper pair of indices
   // TODO

   // Recompute shape and size
   Shape_t shape(sliceSize);
   for (std::size_t i = 0; i < sliceSize; i++) {
      shape[i] = slice[i][1] - slice[i][0];
   }
   auto size = Internal::GetSizeFromShape(shape);

   // Determine first element contributing to the slice and get the data pointer
   Value_t *data;
   Shape_t idx(sliceSize);
   for (std::size_t i = 0; i < sliceSize; i++) {
      idx[i] = slice[i][0];
   }
   data = &this->operator()(idx);

   // Create copy and modify properties
   RTensor<Value_t, Container_t> x(*this);
   x.fData = data;
   x.fShape = shape;
   x.fSize = size;

   // Squeeze tensor and return
   return x.Squeeze();
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
      const auto size = x.GetSize();
      for (std::size_t i = 0; i < size; i++) {
         os << x({i});
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
            os << x({i, j});
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
