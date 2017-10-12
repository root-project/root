// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 19/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////
// Implementation of the TCpuMatrix class. //
/////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu/CpuMatrix.h"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AReal>
TCpuMatrix<AReal>::TCpuMatrix(size_t nRows, size_t nCols)
    : fBuffer(nRows * nCols), fNCols(nCols), fNRows(nRows)
{
   Initialize();
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         (*this)(i, j) = 0;
      }
   }
}

//____________________________________________________________________________
template<typename AReal>
TCpuMatrix<AReal>::TCpuMatrix(const TMatrixT<Double_t> & B)
    : fBuffer(B.GetNoElements()), fNCols(B.GetNcols()), fNRows(B.GetNrows())
{
   Initialize();
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
          (*this)(i,j) = B(i,j);
      }
   }
}

//____________________________________________________________________________
template<typename AReal>
TCpuMatrix<AReal>::TCpuMatrix(const TCpuBuffer<AReal> & buffer,
                               size_t m,
                               size_t n)
    : fBuffer(buffer), fNCols(n), fNRows(m)
{
   Initialize();
}

//____________________________________________________________________________
template<typename AReal>
TCpuMatrix<AReal>::operator TMatrixT<Double_t>() const
{
   TMatrixT<AReal> B(fNRows, fNCols);

   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         B(i,j) = (*this)(i, j);
      }
   }
   return B;
}


//____________________________________________________________________________
template<typename AReal>
void TCpuMatrix<AReal>::Initialize()
{
   if (fNRows > fOnes.size()) {
      fOnes.reserve(fNRows);
      for (size_t i = fOnes.size(); i < fNRows; i++) {
         fOnes.push_back(1.0);
      }
   }
}

// Explicit instantiations.
template class TCpuMatrix<Real_t>;
template class TCpuMatrix<Double_t>;

} // namespace DNN
} // namespace TMVA
