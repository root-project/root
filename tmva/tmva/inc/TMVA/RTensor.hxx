#ifndef ROOT_TMVA_RTENSOR
#define ROOT_TMVA_RTENSOR

#include "ROOT/RVec.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

/// Get size of tensor from shape vector
inline size_t GetSizeFromShape(const std::vector<size_t> &shape)
{
   size_t size = 1;
   for (auto &s : shape)
      size *= s;
   return size;
}

/// Get cumulated shape vector. The cumulation is done from the last element to the first
/// element of the shape vector. This information is needed for the multi-dimensional
/// indexing.
inline std::vector<size_t> GetCumulatedShape(const std::vector<size_t> &shape)
{
   const auto size = shape.size();
   std::vector<size_t> cumulatedShape(size);
   for (size_t i = 0; i < size; i++) {
      if (i == 0) {
         cumulatedShape[size - 1 - i] = 1;
      } else {
         cumulatedShape[size - 1 - i] = cumulatedShape[size - 1 - i + 1] * shape[size - 1 - i + 1];
      }
   }
   return cumulatedShape;
}

} // namespace TMVA::Experimental::Internal

/// RTensor class
template <typename T>
class RTensor {
public:
   // Constructors
   RTensor(const std::vector<size_t> &shape);
   RTensor(T *data, const std::vector<size_t> &shape);

   // Access elements
   T &At(const std::vector<size_t> &idx);
   template <typename... Idx>
   T &operator()(Idx... idx);

   // Shape modifications
   void Reshape(const std::vector<size_t> &shape);

   // Getter
   size_t GetSize() const { return fSize; }
   std::vector<size_t> GetShape() const { return fShape; }
   T *GetData() { return fData.data(); }

private:
   size_t fSize;
   std::vector<size_t> fShape;
   std::vector<size_t> fCumulatedShape;
   ROOT::VecOps::RVec<T> fData;
};

/// Construct a tensor from given shape and initialized with zeros
template <typename T>
RTensor<T>::RTensor(const std::vector<size_t> &shape)
{
   fShape = shape;
   fCumulatedShape = Internal::GetCumulatedShape(shape);
   fSize = Internal::GetSizeFromShape(shape);
   fData = ROOT::VecOps::RVec<T>(fSize);
}

/// Construct a tensor adopting given data
template <typename T>
RTensor<T>::RTensor(T *data, const std::vector<size_t> &shape)
{
   fShape = shape;
   fCumulatedShape = Internal::GetCumulatedShape(shape);
   fSize = Internal::GetSizeFromShape(shape);
   fData = ROOT::VecOps::RVec<T>(data, fSize);
}

/// Reshape tensor
template <typename T>
void RTensor<T>::Reshape(const std::vector<size_t> &shape)
{
   // TODO: assert mul(shape) == mul(new_shape)
   fShape = shape;
   fCumulatedShape = Internal::GetCumulatedShape(shape);
}

/// Access elements with vector of indices
template <typename T>
T &RTensor<T>::At(const std::vector<size_t> &idx)
{
   // TODO: assert idx.size() == fShape.size()
   size_t globalIndex = 0;
   const auto size = idx.size();
   for (size_t i = 0; i < size; i++)
      globalIndex += fCumulatedShape[size - 1 - i] * idx[size - 1 - i];
   return *(fData.data() + globalIndex);
}

/// Access elements with call operator
template <typename T>
template <typename... Idx>
T &RTensor<T>::operator()(Idx... idx)
{
   return this->At({static_cast<size_t>(idx)...});
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
      os << "{ printing not yet implemented for this dimension }";
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
