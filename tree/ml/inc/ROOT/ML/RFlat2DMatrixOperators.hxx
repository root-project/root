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

#include <vector>

// Forward decl
namespace ROOT::Experimental::Internal::ML {
struct RFlat2DMatrix;
} // namespace ROOT::Experimental::Internal::ML

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

   ~RFlat2DMatrixOperators();

   void ShuffleTensor(RFlat2DMatrix &ShuffledTensor, RFlat2DMatrix &Tensor);

   void
   SliceTensor(RFlat2DMatrix &SlicedTensor, RFlat2DMatrix &Tensor, const std::vector<std::vector<std::size_t>> &slice);

   void ConcatenateTensors(RFlat2DMatrix &ConcatTensor, const std::vector<RFlat2DMatrix> &Tensors);
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RFLAT2DMATRIXOPERATORS
