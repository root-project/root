// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDSymEigen
#define ROOT_TMatrixDSymEigen

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSymEigen                                                     //
//                                                                      //
// Eigenvalues and eigenvectors of a real symmetric matrix.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TVectorD.h"

class TMatrixDSymEigen
{
protected :

   static void MakeTridiagonal (TMatrixD &v,TVectorD &d,TVectorD &e);
   static void MakeEigenVectors(TMatrixD &v,TVectorD &d,TVectorD &e);

   TMatrixD fEigenVectors; // Eigen-vectors of matrix
   TVectorD fEigenValues;  // Eigen-values

public :

   enum {kWorkMax = 100}; // size of work array

   TMatrixDSymEigen() : fEigenVectors(), fEigenValues() {};
   TMatrixDSymEigen(const TMatrixDSym      &a);
   TMatrixDSymEigen(const TMatrixDSymEigen &another);
   virtual ~TMatrixDSymEigen() {}

// If matrix A has shape (rowLwb,rowUpb,rowLwb,rowUpb), then each eigen-vector
// must have an index running between (rowLwb,rowUpb) .
// For convenience, the column index of the eigen-vector matrix
// also runs from rowLwb to rowUpb so that the returned matrix
// has also index/shape (rowLwb,rowUpb,rowLwb,rowUpb) .
// The same is true for the eigen-value vector .

   const TMatrixD &GetEigenVectors() const { return fEigenVectors; }
   const TVectorD &GetEigenValues () const { return fEigenValues; }

   TMatrixDSymEigen &operator= (const TMatrixDSymEigen &source);

   ClassDef(TMatrixDSymEigen,1) // Eigen-Vectors/Values of a Matrix
};
#endif
