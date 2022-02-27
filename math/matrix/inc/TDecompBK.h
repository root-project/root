// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Sep 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompBK
#define ROOT_TDecompBK

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Bunch-Kaufman Decomposition class                                     //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "TDecompBase.h"
#include "TMatrixDSym.h"
#include "TVectorD.h"

class TDecompBK : public TDecompBase
{
protected :

   Int_t     fNIpiv;    // size of row permutation index
   Int_t    *fIpiv;     //[fNIpiv] row permutation index
   TMatrixD  fU;        // decomposed matrix so that a = u d u^T

   const TMatrixDBase &GetDecompMatrix() const override { return fU; }

public :

   TDecompBK();
   explicit TDecompBK(Int_t nrows);
   TDecompBK(Int_t row_lwb,Int_t row_upb);
   TDecompBK(const TMatrixDSym &m,Double_t tol = 0.0);
   TDecompBK(const TDecompBK &another);
   ~TDecompBK() override {if (fIpiv) delete [] fIpiv; fIpiv = 0; }

         Int_t     GetNrows  () const override { return fU.GetNrows(); }
         Int_t     GetNcols  () const override { return fU.GetNcols(); }
   const TMatrixD &GetU      ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                                  return fU; }

   virtual       void      SetMatrix (const TMatrixDSym &a);

   Bool_t   Decompose  () override;
   Bool_t   Solve      (      TVectorD &b) override;
   TVectorD Solve      (const TVectorD& b,Bool_t &ok) override { TVectorD x = b; ok = Solve(x); return x; }
   Bool_t   Solve      (      TMatrixDColumn &b) override;
   Bool_t   TransSolve (      TVectorD &b) override            { return Solve(b); }
   TVectorD TransSolve (const TVectorD& b,Bool_t &ok) override { TVectorD x = b; ok = Solve(x); return x; }
   Bool_t   TransSolve (      TMatrixDColumn &b) override      { return Solve(b); }
   void     Det        (Double_t &/*d1*/,Double_t &/*d2*/) override
                                { MayNotUse("Det(Double_t&,Double_t&)"); }

   Bool_t      Invert  (TMatrixDSym &inv);
   TMatrixDSym Invert  (Bool_t &status);
   TMatrixDSym Invert  () { Bool_t status; return Invert(status); }

   void        Print(Option_t *opt ="") const override; // *MENU*

   TDecompBK &operator= (const TDecompBK &source);

   ClassDefOverride(TDecompBK,1) // Matrix Decomposition Bunch-Kaufman
};

#endif
