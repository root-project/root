// @(#)root/matrix:$Name:  $:$Id: TDecompLU.h,v 1.25 2003/09/05 09:21:54 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompLU
#define ROOT_TDecompLU

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// LU Decomposition class                                                //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif

class TDecompLU : public TDecompBase
{
protected :

  Int_t     fNIndex;    // size of row permutation index
  Int_t    *fIndex;     //[fNIndex] row permutation index
  Double_t  fSign;      // = +/- 1 reflecting even/odd row permutations, resp.
  TMatrixD  fLU;        // decomposed matrix so that a = l u where
                        // l is stored lower left and u upper right side

public :

  TDecompLU() {fSign = 0; fIndex = 0; fNIndex = 0;}
  TDecompLU(const TMatrixD &m,Double_t tol = 0.0);
  TDecompLU(      TMatrixD &m,Double_t tol = 0.0);
  TDecompLU(const TDecompLU &another);
  virtual ~TDecompLU() {if (fIndex) delete [] fIndex; fIndex = 0; }

          const TMatrixD      GetMatrix      () const;
  virtual const TMatrixDBase &GetDecompMatrix() const { return fLU; }

  virtual Int_t    Decompose (const TMatrixDBase &a);
  virtual Bool_t   Solve     (TVectorD &b);
  virtual Bool_t   Solve     (TMatrixDColumn &b);
  virtual Bool_t   TransSolve(TVectorD &b);
  virtual Bool_t   TransSolve(TMatrixDColumn &b);
  virtual Double_t Condition ();
  virtual void     Det       (Double_t &d1,Double_t &d2);

  static  Int_t  DecomposeLU(TMatrixD &lu,Int_t *index,Double_t &sign,
                             Double_t tol,Int_t &nrZeros);
  static  Int_t  InvertLU   (TMatrixD &lu,Int_t *index,Double_t tol);

  TDecompLU &operator= (const TDecompLU &source);

  ClassDef(TDecompLU,1) // Matrix Decompositition LU
};

#endif
