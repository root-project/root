// @(#)root/matrix:$Name:  $:$Id: TDecompChol.h,v 1.3 2004/02/04 17:12:44 brun Exp $
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

  virtual const TMatrixD &GetDecompMatrix() const { return fU; }

public :

  TDecompChol() {};
  TDecompChol(const TMatrixDSym &a,Double_t tol = 0.0);
  TDecompChol(const TMatrixD    &a,Double_t tol = 0.0);
  TDecompChol(const TDecompChol &another);
  virtual ~TDecompChol() {}

          const TMatrixD  GetMatrix () const;
  virtual       Int_t     GetNrows  () const { return fU.GetNrows(); }
  virtual       Int_t     GetNcols  () const { return fU.GetNcols(); }
          const TMatrixD &GetU      () const { return fU; }

  virtual Int_t    Decompose  () { Error("Decompose","Use Decompose(const TMatrixD&)"); return kFALSE; }
          Int_t    Decompose  (const TMatrixD &a);
  virtual Bool_t   Solve      (      TVectorD &b);
  virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok);
  virtual Bool_t   Solve      (      TMatrixDColumn &b);
  virtual Bool_t   TransSolve (      TVectorD &b)            { return Solve(b); }
  virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
  virtual Bool_t   TransSolve (      TMatrixDColumn &b)      { return Solve(b); }
  virtual void     Det        (Double_t &d1,Double_t &d2);

  TDecompChol &operator= (const TDecompChol &source);

  ClassDef(TDecompChol,1) // Matrix Decompositition Cholesky
};

TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b);
TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b,const TVectorD &std);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &b);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B,const TVectorD &std);

#endif
