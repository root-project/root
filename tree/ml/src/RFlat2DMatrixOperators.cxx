#include "RFlat2DMatrixOperators.hxx"

#include <algorithm>
#include <numeric>
#include <random>

#include "ROOT/ML/RFlat2DMatrix.hxx"

namespace ROOT::Experimental::Internal::ML {

RFlat2DMatrixOperators::~RFlat2DMatrixOperators() = default;

std::vector<std::size_t> RFlat2DMatrixOperators::MakePermutation(std::size_t rows)
{
   if (!fShuffle) {
      return {};
   }

   // make an identity permutation map
   std::vector<std::size_t> indices(rows);
   std::iota(indices.begin(), indices.end(), 0);

   // shuffle the identity permutation to create a new permutation
   ShuffleIndices(indices);

   return indices;
}

void RFlat2DMatrixOperators::ShuffleIndices(std::vector<std::size_t> &indices)
{
   if (!fShuffle) {
      return;
   }

   std::random_device rd;
   std::mt19937 g;

   if (fSetSeed == 0) {
      g.seed(rd());
   } else {
      g.seed(fSetSeed);
   }

   std::shuffle(indices.begin(), indices.end(), g);
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

} // namespace ROOT::Experimental::Internal::ML
