#include "ROOT/ML/RFlat2DMatrixOperators.hxx"

#include <algorithm>
#include <numeric>
#include <random>

#include "Rtypes.h"
#include "ROOT/ML/RFlat2DMatrix.hxx"

namespace ROOT::Experimental::Internal::ML {

RFlat2DMatrixOperators::~RFlat2DMatrixOperators() = default;

void RFlat2DMatrixOperators::ShuffleTensor(RFlat2DMatrix &ShuffledTensor, RFlat2DMatrix &Tensor)
{
   if (fShuffle) {
      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      std::size_t rows = Tensor.GetRows();
      std::size_t cols = Tensor.GetCols();
      ShuffledTensor.Resize(rows, cols);

      // make an identity permutation map
      std::vector<Long_t> indices(rows);
      std::iota(indices.begin(), indices.end(), 0);

      // shuffle the identity permutation to create a new permutation
      std::shuffle(indices.begin(), indices.end(), g);

      // shuffle data in the tensor with the permutation map defined above
      for (std::size_t i = 0; i < rows; i++) {
         std::copy(Tensor.GetData() + indices[i] * cols, Tensor.GetData() + (indices[i] + 1) * cols,
                   ShuffledTensor.GetData() + i * cols);
      }
   } else {
      ShuffledTensor = Tensor;
   }
}

void RFlat2DMatrixOperators::SliceTensor(RFlat2DMatrix &SlicedTensor, RFlat2DMatrix &Tensor,
                                         const std::vector<std::vector<std::size_t>> &slice)
{
   const auto &rowSlice = slice[0];
   const auto &colSlice = slice[1];

   std::size_t rowStart = rowSlice[0];
   std::size_t rowEnd = rowSlice[1];
   std::size_t colStart = colSlice[0];
   std::size_t colEnd = colSlice[1];

   std::size_t rows = rowEnd - rowStart;
   std::size_t cols = colEnd - colStart;

   SlicedTensor.Resize(rows, cols);
   std::copy(Tensor.GetData() + rowStart * cols, Tensor.GetData() + rowStart * cols + rows * cols,
             SlicedTensor.GetData());
}

void RFlat2DMatrixOperators::ConcatenateTensors(RFlat2DMatrix &ConcatTensor, const std::vector<RFlat2DMatrix> &Tensors)
{
   std::size_t cols = Tensors[0].GetCols();
   std::size_t rows = 0;

   for (const auto &t : Tensors) {
      rows += t.GetRows();
   }

   ConcatTensor.Resize(rows, cols);

   std::size_t index = 0;
   for (std::size_t i = 0; i < Tensors.size(); i++) {
      std::size_t tensorRows = Tensors[i].GetRows();
      std::copy(Tensors[i].GetData(), Tensors[i].GetData() + tensorRows * cols, ConcatTensor.GetData() + index * cols);
      index += tensorRows;
   }
}
} // namespace ROOT::Experimental::Internal::ML
