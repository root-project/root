// @(#)root/matrix:$Name:  $:$Id: TMatrixDSymEigen.h,v 1.1 2004/01/25 20:33:32 brun Exp $
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

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

class TMatrixDSymEigen
{
protected :

  static void MakeTridiagonal (TMatrixD &v,TVectorD &d,TVectorD &e);
  static void MakeEigenVectors(TMatrixD &v,TVectorD &d,TVectorD &e);

  TMatrixD fEigenVectors; // Eigen-vectors of matrix
  TVectorD fEigenValues;  // Eigen-values

public :

  enum {kWorkMax = 100}; // size of work array

  TMatrixDSymEigen() {};
  TMatrixDSymEigen(const TMatrixDSym      &a);
  TMatrixDSymEigen(const TMatrixDSymEigen &another);
  virtual ~TMatrixDSymEigen() {}

  const TMatrixD &GetEigenVectors() const { return fEigenVectors; }
  const TVectorD &GetEigenValues () const { return fEigenValues; }

  TMatrixDSymEigen &operator= (const TMatrixDSymEigen &source);

  ClassDef(TMatrixDSymEigen,1) // Eigen-Vectors/Values of a Matrix
};
#endif
