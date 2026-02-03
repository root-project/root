// Author: Martin Føll, University of Oslo (UiO) & CERN 1/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RFLAT2DMATRIXOPERATORS
#define ROOT_INTERNAL_ML_RFLAT2DMATRIXOPERATORS

#include <random>
#include <algorithm>

#include "ROOT/ML/RFlat2DMatrix.hxx"

namespace ROOT::Experimental::Internal::ML {
/**
\class ROOT::Experimental::Internal::ML::RFlat2DMatrixOperators

\brief Collection of operations applied to one or multiple flat 2D matrices.
*/

class RFlat2DMatrixOperators {
private:
   bool fShuffle;
   std::size_t fSetSeed;

public:
   RFlat2DMatrixOperators(bool shuffle = true, const std::size_t setSeed = 0) : fShuffle(shuffle), fSetSeed(setSeed) {}

   void ShuffleTensor(RFlat2DMatrix &ShuffledTensor, RFlat2DMatrix &Tensor)
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

   void
   SliceTensor(RFlat2DMatrix &SlicedTensor, RFlat2DMatrix &Tensor, const std::vector<std::vector<std::size_t>> &slice)
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

   void ConcatenateTensors(RFlat2DMatrix &ConcatTensor, const std::vector<RFlat2DMatrix> &Tensors)
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
         std::copy(Tensors[i].GetData(), Tensors[i].GetData() + tensorRows * cols,
                   ConcatTensor.GetData() + index * cols);
         index += tensorRows;
      }
   }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RFLAT2DMATRIXOPERATORS
