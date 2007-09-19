// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompQRH
#define ROOT_TDecompQRH

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// QR Decomposition class                                                //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif

class TDecompQRH : public TDecompBase
{
protected :

   //  A = fQ fR H (m x n) matrix
   TMatrixD fQ;  // (m x n) - orthogonal matrix
   TMatrixD fR;  // (n x n) - upper triangular matrix
   TVectorD fUp; // (n) - vector with Householder up's
   TVectorD fW;  // (n) - vector with Householder beta's

   static Bool_t QRH(TMatrixD &q,TVectorD &diagR,TVectorD &up,TVectorD &w,Double_t tol);

   virtual const TMatrixDBase &GetDecompMatrix() const { return fR; }

public :

   enum {kWorkMax = 100}; // size of work array

   TDecompQRH() {}
   TDecompQRH(Int_t nrows,Int_t ncols);
   TDecompQRH(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
   TDecompQRH(const TMatrixD &m,Double_t tol = 0.0); // be careful for slicing in operator=
   TDecompQRH(const TDecompQRH &another);
   virtual ~TDecompQRH() {}

   virtual       Int_t     GetNrows () const { return fQ.GetNrows(); }
   virtual       Int_t     GetNcols () const { return fQ.GetNcols(); }
   virtual const TMatrixD &GetQ     ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                               return fQ; }
   virtual const TMatrixD &GetR     ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                               return fR; }
   virtual const TVectorD &GetUp    ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                               return fUp; }
   virtual const TVectorD &GetW     ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                               return fW; }

   virtual       void      SetMatrix(const TMatrixD &a);

   virtual Bool_t   Decompose  ();
   virtual Bool_t   Solve      (      TVectorD &b);
   virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   Solve      (      TMatrixDColumn &b);
   virtual Bool_t   TransSolve (      TVectorD &b);
   virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = TransSolve(x); return x; }
   virtual Bool_t   TransSolve (      TMatrixDColumn &b);
   virtual void     Det        (Double_t &d1,Double_t &d2);

           Bool_t   Invert     (TMatrixD &inv);
           TMatrixD Invert     (Bool_t &status);
           TMatrixD Invert     () { Bool_t status; return Invert(status); }

   void Print(Option_t *opt ="") const; // *MENU*

   TDecompQRH &operator= (const TDecompQRH &source);

   ClassDef(TDecompQRH,1) // Matrix Decompositition QRH
};

#endif
