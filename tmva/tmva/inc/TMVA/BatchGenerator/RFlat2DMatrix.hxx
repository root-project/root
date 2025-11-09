#ifndef ROOT_TMVA_RFLAT2DMATRIX
#define ROOT_TMVA_RFLAT2DMATRIX

#include <utility>
#include <cassert>

#include "ROOT/RVec.hxx"

namespace TMVA::Experimental::Internal {
/// \brief Wrapper around ROOT::RVec<float> representing a 2D matrix
///
/// The storage is flattened row-major: index(row, col) == row * cols + col.
struct RFlat2DMatrix {
   ROOT::RVecF fRVec;
   std::size_t fRows{0};
   std::size_t fCols{0};

   RFlat2DMatrix() = default;

   RFlat2DMatrix(std::size_t rows, std::size_t cols) { Resize(rows, cols); }

   float *GetData() { return fRVec.data(); }

   const float *GetData() const { return fRVec.data(); }

   // Used in the pythonization
   std::pair<std::size_t, std::size_t> GetShape() const { return {fRows, fCols}; }

   std::size_t GetRows() const { return fRows; }

   std::size_t GetCols() const { return fCols; }

   std::size_t GetSize() const { return fRVec.size(); }

   void Resize(std::size_t rows, std::size_t cols)
   {
      fRows = rows;
      fCols = cols;
      fRVec.resize(rows * cols);
   }

   void Reshape(std::size_t rows, std::size_t cols)
   {
      // We don't reallocate: require matching sizes
      assert(rows * cols == fRVec.size());
      fRows = rows;
      fCols = cols;
   }

   float &operator[](std::size_t i) { return fRVec[i]; }

   const float &operator[](std::size_t i) const { return fRVec[i]; }
};

} // namespace TMVA::Experimental::Internal
#endif // ROOT_TMVA_RFLAT2DMATRIX
