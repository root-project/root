// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompSVD
#define ROOT_TDecompSVD

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Single Value Decomposition class                                      //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif

class TDecompSVD : public TDecompBase
{
protected :

   //  A = fU fSig fV^T
   TMatrixD fU;    // orthogonal matrix
   TMatrixD fV;    // orthogonal matrix
   TVectorD fSig;  // diagonal of diagonal matrix

   static Bool_t Bidiagonalize(TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag);
   static Bool_t Diagonalize  (TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag);
   static void   Diag_1       (TMatrixD &v,TVectorD &sDiag,TVectorD &oDiag,Int_t k);
   static void   Diag_2       (TVectorD &sDiag,TVectorD &oDiag,Int_t k,Int_t l);
   static void   Diag_3       (TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag,Int_t k,Int_t l);
   static void   SortSingular (TMatrixD &v,TMatrixD &u,TVectorD &sDiag);

   virtual const TMatrixDBase &GetDecompMatrix() const { return fU; }

public :

   enum {kWorkMax = 100}; // size of work array

   TDecompSVD(): fU(), fV(), fSig() {}
   TDecompSVD(Int_t nrows,Int_t ncols);
   TDecompSVD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
   TDecompSVD(const TMatrixD &m,Double_t tol = 0.0);
   TDecompSVD(const TDecompSVD &another);
   virtual ~TDecompSVD() {}

           const TMatrixD  GetMatrix ();
   Int_t     GetNrows  () const { return fU.GetNrows(); }
   Int_t     GetNcols  () const { return fV.GetNcols(); }
           const TMatrixD &GetU      ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                                 return fU; }
           const TMatrixD &GetV      ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                                return fV; }
           const TVectorD &GetSig    ()       { if ( !TestBit(kDecomposed) ) Decompose();
                                                return fSig; }

   virtual       void      SetMatrix (const TMatrixD &a);

   virtual Bool_t   Decompose  ();
   virtual Bool_t   Solve      (      TVectorD &b);
   virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); 
                                                                const Int_t rowLwb = GetRowLwb();
                                                                x.ResizeTo(rowLwb,rowLwb+GetNcols()-1);
                                                                return x; }
   virtual Bool_t   Solve      (      TMatrixDColumn &b);
   virtual Bool_t   TransSolve (      TVectorD &b);
   virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = TransSolve(x);
                                                                const Int_t rowLwb = GetRowLwb();
                                                                x.ResizeTo(rowLwb,rowLwb+GetNcols()-1);
                                                                return x; }
   virtual Bool_t   TransSolve (      TMatrixDColumn &b);
   virtual Double_t Condition  ();
   virtual void     Det        (Double_t &d1,Double_t &d2);

           Bool_t   Invert     (TMatrixD &inv);
           TMatrixD Invert     (Bool_t &status);
           TMatrixD Invert     () {Bool_t status; return Invert(status); }

   void Print(Option_t *opt ="") const; // *MENU*

   TDecompSVD &operator= (const TDecompSVD &source);

   ClassDef(TDecompSVD,1) // Matrix Decompositition SVD
};

#endif
