// @(#)root/matrix:$Id$
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

#include "TDecompBase.h"
#include "TMatrixDSym.h"

class TDecompChol : public TDecompBase
{
protected :

   TMatrixD fU; // decomposed matrix fU so that a = fU^T fU

   const TMatrixDBase &GetDecompMatrix() const override { return fU; }

public :

   TDecompChol() : fU() {}
   explicit TDecompChol(Int_t nrows);
   TDecompChol(Int_t row_lwb,Int_t row_upb);
   TDecompChol(const TMatrixDSym &a,Double_t tol = 0.0);
   TDecompChol(const TMatrixD    &a,Double_t tol = 0.0);
   TDecompChol(const TDecompChol &another);
   ~TDecompChol() override {}

           const TMatrixDSym GetMatrix ();
         Int_t       GetNrows  () const override { return fU.GetNrows(); }
         Int_t       GetNcols  () const override { return fU.GetNcols(); }
           const TMatrixD   &GetU      () const { return fU; }

   virtual       void        SetMatrix (const TMatrixDSym &a);

   Bool_t   Decompose  () override;
   Bool_t   Solve      (      TVectorD &b) override;
   TVectorD Solve      (const TVectorD& b,Bool_t &ok) override { TVectorD x = b; ok = Solve(x); return x; }
   Bool_t   Solve      (      TMatrixDColumn &b) override;
   Bool_t   TransSolve (      TVectorD &b) override            { return Solve(b); }
   TVectorD TransSolve (const TVectorD& b,Bool_t &ok) override { TVectorD x = b; ok = Solve(x); return x; }
   Bool_t   TransSolve (      TMatrixDColumn &b) override      { return Solve(b); }
   void     Det        (Double_t &d1,Double_t &d2) override;

           Bool_t      Invert  (TMatrixDSym &inv);
           TMatrixDSym Invert  (Bool_t &status);
           TMatrixDSym Invert  () { Bool_t status; return Invert(status); }

   void Print(Option_t *opt ="") const override; // *MENU*

   TDecompChol &operator= (const TDecompChol &source);

   ClassDefOverride(TDecompChol,2) // Matrix Decompositition Cholesky
};

TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b);
TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b,const TVectorD &std);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &b);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B,const TVectorD &std);

#endif
