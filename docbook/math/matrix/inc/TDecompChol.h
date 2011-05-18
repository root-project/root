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

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif
#ifndef ROOT_TMatrixDSym
#include "TMatrixDSym.h"
#endif

class TDecompChol : public TDecompBase
{
protected :

   TMatrixD fU; // decomposed matrix fU so that a = fU^T fU

   virtual const TMatrixDBase &GetDecompMatrix() const { return fU; }

public :

   TDecompChol() : fU() {}
   explicit TDecompChol(Int_t nrows);
   TDecompChol(Int_t row_lwb,Int_t row_upb);
   TDecompChol(const TMatrixDSym &a,Double_t tol = 0.0);
   TDecompChol(const TMatrixD    &a,Double_t tol = 0.0);
   TDecompChol(const TDecompChol &another);
   virtual ~TDecompChol() {}

           const TMatrixDSym GetMatrix ();
   virtual       Int_t       GetNrows  () const { return fU.GetNrows(); }
   virtual       Int_t       GetNcols  () const { return fU.GetNcols(); }
           const TMatrixD   &GetU      () const { return fU; }

   virtual       void        SetMatrix (const TMatrixDSym &a);

   virtual Bool_t   Decompose  ();
   virtual Bool_t   Solve      (      TVectorD &b);
   virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   Solve      (      TMatrixDColumn &b);
   virtual Bool_t   TransSolve (      TVectorD &b)            { return Solve(b); }
   virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   TransSolve (      TMatrixDColumn &b)      { return Solve(b); }
   virtual void     Det        (Double_t &d1,Double_t &d2);

           Bool_t      Invert  (TMatrixDSym &inv);
           TMatrixDSym Invert  (Bool_t &status);
           TMatrixDSym Invert  () { Bool_t status; return Invert(status); }

   void Print(Option_t *opt ="") const; // *MENU*

   TDecompChol &operator= (const TDecompChol &source);

   ClassDef(TDecompChol,2) // Matrix Decompositition Cholesky
};

TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b);
TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b,const TVectorD &std);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &b);
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B,const TVectorD &std);

#endif
