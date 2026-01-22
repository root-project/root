// Author: Martin Føll, University of Oslo (UiO) & CERN 1/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RFLAT2DMATRIXOPERATORS
#define TMVA_RFLAT2DMATRIXOPERATORS

#include <random>
#include <algorithm>

#include "TMVA/BatchGenerator/RFlat2DMatrix.hxx"

namespace TMVA::Experimental::Internal {
// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RFlat2DMatrixOperators
\ingroup tmva
\brief Collection of operations applied to one or multiple flat 2D matrices.
*/

class RFlat2DMatrixOperators {
private:
   // clang-format on   
   bool fShuffle;
   std::size_t fSetSeed;   
public:
   RFlat2DMatrixOperators(bool shuffle = true, const std::size_t setSeed = 0)
      : fShuffle(shuffle),
        fSetSeed(setSeed)
   {
   
   }

  void ShuffleTensor(RFlat2DMatrix &ShuffledTensor, RFlat2DMatrix &Tensor ) {
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
    if (fShuffle) {
      std::shuffle(indices.begin(), indices.end(), g);
    }

    // shuffle data in the tensor with the permutation map defined above
    for (std::size_t i = 0; i < rows; i++) {
      std::copy(Tensor.GetData() + indices[i] * cols,
                Tensor.GetData() + (indices[i] + 1) * cols,
                ShuffledTensor.GetData() + i * cols);
    }
  }
   
};

} // namespace TMVA::Experimental::Internal
#endif // ROOT_TMVA_RFLAT2DMATRIXOPERATORS
