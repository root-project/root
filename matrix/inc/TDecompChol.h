// @(#)root/matrix:$Name:  $:$Id: TDecompChol.h,v 1.25 2003/09/05 09:21:54 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompChol
#define ROOT_TDecompChol

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Cholesky Decomposition class                                          //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif

class TDecompChol : public TDecompBase
{
protected :

  TMatrixD fU; // decomposed matrix fU so that a = fU^T fU

public :

  TDecompChol() {};
  TDecompChol(const TMatrixDSym &a,Double_t tol = 0.0);
  TDecompChol(const TMatrixD    &a,Double_t tol = 0.0);
  TDecompChol(const TDecompChol &another);
  virtual ~TDecompChol() {}

          const TMatrixD      GetMatrix      () const;
  virtual const TMatrixDBase &GetDecompMatrix() const { return fU; }

  virtual Int_t  Decompose (const TMatrixDBase &a);
  virtual Bool_t Solve     (TVectorD &b);
  virtual Bool_t Solve     (TMatrixDColumn &b);
  virtual Bool_t TransSolve(TVectorD &b)       { return Solve(b); }
  virtual Bool_t TransSolve(TMatrixDColumn &b) { return Solve(b); }
  virtual void   Det       (Double_t &d1,Double_t &d2);

  TDecompChol &operator= (const TDecompChol &source);

  ClassDef(TDecompChol,1) // Matrix Decompositition Cholesky
};

TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &b);

#endif
