#ifndef ROOT_TMVA_RTENSOR
#define ROOT_TMVA_RTENSOR

#include "ROOT/RVec.hxx"

#include <sstream>
#include <cstdint>

namespace TMVA {
namespace Experimental {

/// Memory order types
struct MemoryOrder {
   static const uint8_t RowMajor = 0;
   static const uint8_t ColumnMajor = 1;
};

namespace Internal {

/// Get size of tensor from shape vector
inline size_t GetSizeFromShape(const std::vector<size_t> &shape)
{
   if (shape.size() == 0)
      return 0;
   size_t size = 1;
   for (auto &s : shape)
      size *= s;
   return size;
}

/// Compute strides from shape vector.
/// This information is needed for the multi-dimensional indexing. See here:
/// https://en.wikipedia.org/wiki/Row-_and_column-major_order
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
inline std::vector<size_t> ComputeStrides(const std::vector<size_t> &shape, uint8_t memoryOrder)
{
   const auto size = shape.size();
   std::vector<size_t> strides(size);
   if (memoryOrder == MemoryOrder::RowMajor) {
      for (size_t i = 0; i < size; i++) {
         if (i == 0) {
            strides[size - 1 - i] = 1;
         } else {
            strides[size - 1 - i] = strides[size - 1 - i + 1] * shape[size - 1 - i + 1];
         }
      }
   } else if (memoryOrder == MemoryOrder::ColumnMajor) {
      for (size_t i = 0; i < size; i++) {
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

} // namespace TMVA::Experimental::Internal

/// RTensor class
template <typename T>
class RTensor {
public:
   // Constructors
   RTensor(const std::vector<size_t> &shape, uint8_t memoryOrder = MemoryOrder::RowMajor);
   RTensor(T *data, const std::vector<size_t> &shape, uint8_t memoryOrder = MemoryOrder::RowMajor);

   // Access elements
   T &At(const std::vector<size_t> &idx);
   template <typename... Idx>
   T &operator()(Idx... idx);

   // Shape modifications
   void Reshape(const std::vector<size_t> &shape);

   // Get properties of container
   size_t GetSize() const { return fSize; }                    // Return size of contiguous memory
   std::vector<size_t> GetShape() const { return fShape; }     // Return shape
   std::vector<size_t> GetStrides() const { return fStrides; } // Return strides
   T *GetData() { return fData.data(); }                       // Return pointer to data
   ROOT::VecOps::RVec<T> &GetDataAsVec() { return fData; }     // Return data as reference to vector
   uint8_t GetMemoryOrder() { return fMemoryOrder; }           // Return memory order

private:
   size_t fSize;                 // Size of contiguous memory
   std::vector<size_t> fShape;   // Shape of tensor
   std::vector<size_t> fStrides; // Strides, needed for indexing
   ROOT::VecOps::RVec<T> fData;  // Container for contiguous memory
   uint8_t fMemoryOrder;         // Memory ordering
};

/// Construct a tensor from given shape initialized with zeros
template <typename T>
RTensor<T>::RTensor(const std::vector<size_t> &shape, uint8_t memoryOrder)
{
   fShape = shape;
   fStrides = Internal::ComputeStrides(shape, memoryOrder);
   fSize = Internal::GetSizeFromShape(shape);
   fData = ROOT::VecOps::RVec<T>(fSize);
   fMemoryOrder = memoryOrder;
}

/// Construct a tensor adopting given data
template <typename T>
RTensor<T>::RTensor(T *data, const std::vector<size_t> &shape, uint8_t memoryOrder)
{
   fShape = shape;
   fStrides = Internal::ComputeStrides(shape, memoryOrder);
   fSize = Internal::GetSizeFromShape(shape);
   fData = ROOT::VecOps::RVec<T>(data, fSize);
   fMemoryOrder = memoryOrder;
}

/// Reshape tensor
template <typename T>
void RTensor<T>::Reshape(const std::vector<size_t> &shape)
{
   const auto sizeInput = Internal::GetSizeFromShape(shape);
   if (sizeInput != fSize) {
      std::stringstream ss;
      ss << "Cannot reshape tensor with size " << fSize << " into shape { ";
      for (size_t i = 0; i < shape.size(); i++) {
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

/// Access elements with vector of indices
/// The indices are checked whether they match the shape.
template <typename T>
T &RTensor<T>::At(const std::vector<size_t> &idx)
{
   const auto size = idx.size();
   if (size != fShape.size()) {
      std::stringstream ss;
      ss << "Number of indices (" << size << ") do not match number of dimensions (" << fShape.size() << ").";
      throw std::runtime_error(ss.str());
   }
   size_t globalIndex = 0;
   for (size_t i = 0; i < size; i++) {
      globalIndex += fStrides[i] * idx[i];
   }
   return *(fData.data() + globalIndex);
}

/// Access elements with call operator
/// The indices are not checked whether they match the shape.
template <typename T>
template <typename... Idx>
T &RTensor<T>::operator()(Idx... args)
{
   const std::vector<size_t> idx = {static_cast<size_t>(args)...};
   size_t globalIndex = 0;
   const auto size = idx.size();
   for (size_t i = 0; i < size; i++) {
      globalIndex += fStrides[i] * idx[i];
   }
   return *(fData.data() + globalIndex);
}

/// Pretty printing
template <typename T>
std::ostream &operator<<(std::ostream &os, RTensor<T> &x)
{
   const auto shapeSize = x.GetShape().size();
   if (shapeSize == 1) {
      os << "{ ";
      auto data = x.GetData();
      const auto size = x.GetSize();
      for (size_t i = 0; i < size; i++) {
         os << *(data + i);
         if (i != size - 1)
            os << ", ";
      }
      os << " }";
   } else if (shapeSize == 2) {
      os << "{";
      const auto shape = x.GetShape();
      for (size_t i = 0; i < shape[0]; i++) {
         os << " { ";
         for (size_t j = 0; j < shape[1]; j++) {
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
inline std::string printValue(TMVA::Experimental::RTensor<T> *x)
{
   std::stringstream ss;
   ss << *x;
   return ss.str();
}
} // namespace cling

#endif // ROOT_TMVA_RTENSOR
