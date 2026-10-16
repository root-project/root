#include "RFlat2DMatrixOperators.hxx"

#include <algorithm>
#include <numeric>
#include <random>

#include "ROOT/ML/RFlat2DMatrix.hxx"

namespace ROOT::Experimental::Internal::ML {

RFlat2DMatrixOperators::~RFlat2DMatrixOperators() = default;

RFlat2DMatrix &RFlat2DMatrixOperators::ShuffleTensor(RFlat2DMatrix &ShuffledTensor, RFlat2DMatrix &Tensor)
{
   const auto permutation = MakePermutation(Tensor.GetRows());
   // nothing to permute, no copy
   if (permutation.empty()) {
      return Tensor;
   }

   ShuffledTensor.fRVec.clear();
   ShuffledTensor.fRVec.reserve(Tensor.GetSize());
   AppendRows(ShuffledTensor, Tensor, permutation, 0, Tensor.GetRows());

   return ShuffledTensor;
}

std::vector<std::size_t> RFlat2DMatrixOperators::MakePermutation(std::size_t rows)
{
   if (!fShuffle) {
      return {};
   }

   std::random_device rd;
   std::mt19937 g;

   if (fSetSeed == 0) {
      g.seed(rd());
   } else {
      g.seed(fSetSeed);
   }

   // make an identity permutation map
   std::vector<std::size_t> indices(rows);
   std::iota(indices.begin(), indices.end(), 0);

   // shuffle the identity permutation to create a new permutation
   std::shuffle(indices.begin(), indices.end(), g);

   return indices;
}

void RFlat2DMatrixOperators::AppendRows(RFlat2DMatrix &Dest, const RFlat2DMatrix &Tensor,
                                        const std::vector<std::size_t> &permutation, std::size_t begin, std::size_t end)
{
   if (begin == end) {
      return;
   }

   const std::size_t cols = Tensor.GetCols();
   const float *data = Tensor.GetData();
   auto &values = Dest.fRVec;

   // grow once and copy the rows in place, cheaper than inserting them one by one
   const std::size_t offset = values.size();
   values.resize(offset + (end - begin) * cols);
   float *out = values.data() + offset;

   if (permutation.empty()) {
      std::copy(data + begin * cols, data + end * cols, out);
   } else {
      for (std::size_t i = begin; i < end; i++) {
         out = std::copy_n(data + permutation[i] * cols, cols, out);
      }
   }

   Dest.Reshape(values.size() / cols, cols);
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
