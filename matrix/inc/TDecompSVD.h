// @(#)root/matrix:$Name:  $:$Id: TDecompSVD.h,v 1.5 2004/02/12 13:03:00 brun Exp $
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

  virtual const TMatrixD &GetDecompMatrix() const { return fU; }

public :

  enum {kWorkMax = 100}; // size of work array

  TDecompSVD() {}
  TDecompSVD(Int_t nrows,Int_t ncols);
  TDecompSVD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TDecompSVD(const TMatrixD &m,Double_t tol = 0.0);
  TDecompSVD(const TDecompSVD &another);
  virtual ~TDecompSVD() {}

          const TMatrixD  GetMatrix ();
  virtual       Int_t     GetNrows  () const { return fU.GetNrows(); }
  virtual       Int_t     GetNcols  () const { return fV.GetNcols(); }
          const TMatrixD &GetU      ()       { if ( !( fStatus & kDecomposed ) ) Decompose();
                                               return fU; }
          const TMatrixD &GetV      ()       { if ( !( fStatus & kDecomposed ) ) Decompose();
                                               return fV; }
          const TVectorD &GetSig    ()       { if ( !( fStatus & kDecomposed ) ) Decompose();
                                               return fSig; }

  virtual       void      SetMatrix (const TMatrixD &a);

  virtual Int_t    Decompose  ();
  virtual Bool_t   Solve      (      TVectorD &b);
  virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok);
  virtual Bool_t   Solve      (      TMatrixDColumn &b);
  virtual Bool_t   TransSolve (      TVectorD &b);
  virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok);
  virtual Bool_t   TransSolve (      TMatrixDColumn &b);
  virtual Double_t Condition  ();
  virtual void     Det        (Double_t &d1,Double_t &d2);

  TDecompSVD &operator= (const TDecompSVD &source);

  ClassDef(TDecompSVD,1) // Matrix Decompositition SVD
};

#endif
