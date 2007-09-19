// @(#)root/matrix:$Id$
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

   Int_t     fImplicitPivot; // control to determine implicit row scale before
                             //  deciding on the pivot (Crout method)
   Int_t     fNIndex;        // size of row permutation index
   Int_t    *fIndex;         //[fNIndex] row permutation index
   Double_t  fSign;          // = +/- 1 reflecting even/odd row permutations, resp.
   TMatrixD  fLU;            // decomposed matrix so that a = l u where
                             // l is stored lower left and u upper right side

   static Bool_t DecomposeLUCrout(TMatrixD &lu,Int_t *index,Double_t &sign,Double_t tol,Int_t &nrZeros);
   static Bool_t DecomposeLUGauss(TMatrixD &lu,Int_t *index,Double_t &sign,Double_t tol,Int_t &nrZeros);

   virtual const TMatrixDBase &GetDecompMatrix() const { return fLU; }

public :

   TDecompLU();
   explicit TDecompLU(Int_t nrows);
   TDecompLU(Int_t row_lwb,Int_t row_upb);
   TDecompLU(const TMatrixD &m,Double_t tol = 0.0,Int_t implicit = 1);
   TDecompLU(const TDecompLU &another);
   virtual ~TDecompLU() {if (fIndex) delete [] fIndex; fIndex = 0; }

           const TMatrixD  GetMatrix ();
   virtual       Int_t     GetNrows  () const { return fLU.GetNrows(); }
   virtual       Int_t     GetNcols  () const { return fLU.GetNcols(); }
           const TMatrixD &GetLU     ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                                return fLU; }

   virtual       void      SetMatrix (const TMatrixD &a);

   virtual Bool_t   Decompose  ();
   virtual Bool_t   Solve      (      TVectorD &b);
   virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   Solve      (      TMatrixDColumn &b);
   virtual Bool_t   TransSolve (      TVectorD &b);
   virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = TransSolve(x); return x; }
   virtual Bool_t   TransSolve (      TMatrixDColumn &b);
   virtual void     Det        (Double_t &d1,Double_t &d2);

   static  Bool_t   InvertLU  (TMatrixD &a,Double_t tol,Double_t *det=0);
   Bool_t           Invert    (TMatrixD &inv);
   TMatrixD         Invert    (Bool_t &status);
   TMatrixD         Invert    () { Bool_t status; return Invert(status); }

   void Print(Option_t *opt ="") const; // *MENU*

   TDecompLU &operator= (const TDecompLU &source);

   ClassDef(TDecompLU,1) // Matrix Decompositition LU
};

#endif
