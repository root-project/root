// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDEigen
#define ROOT_TMatrixDEigen

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDEigen                                                        //
//                                                                      //
// Eigenvalues and eigenvectors of a real matrix.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif

class TMatrixDEigen
{
protected :

   static void MakeHessenBerg  (TMatrixD &v,TVectorD &ortho,TMatrixD &H);
   static void MakeSchurr      (TMatrixD &v,TVectorD &d,    TVectorD &e,TMatrixD &H);
   static void Sort            (TMatrixD &v,TVectorD &d,    TVectorD &e);

   TMatrixD fEigenVectors;   // Eigen-vectors of matrix
   TVectorD fEigenValuesRe;  // Eigen-values
   TVectorD fEigenValuesIm;  // Eigen-values

public :

   enum {kWorkMax = 100}; // size of work array

   TMatrixDEigen() 
     : fEigenVectors(), fEigenValuesRe(), fEigenValuesIm() {};
   TMatrixDEigen(const TMatrixD &a);
   TMatrixDEigen(const TMatrixDEigen &another);
   virtual ~TMatrixDEigen() {}

// If matrix A has shape (rowLwb,rowUpb,rowLwb,rowUpb), then each eigen-vector
// must have an index running between (rowLwb,rowUpb) .
// For convenience, the column index of the eigen-vector matrix
// also runs from rowLwb to rowUpb so that the returned matrix
// has also index/shape (rowLwb,rowUpb,rowLwb,rowUpb) .
// The same is true for the eigen-value vectors an matrix .

   const TMatrixD &GetEigenVectors () const { return fEigenVectors;  }
   const TVectorD &GetEigenValuesRe() const { return fEigenValuesRe; }
   const TVectorD &GetEigenValuesIm() const { return fEigenValuesIm; }
   const TMatrixD  GetEigenValues  () const;

   TMatrixDEigen &operator= (const TMatrixDEigen &source);

   ClassDef(TMatrixDEigen,1) // Eigen-Vectors/Values of a Matrix
};
#endif
