// @(#)root/matrix:$Name:  $:$Id: TDecompQRH.h,v 1.25 2003/09/05 09:21:54 brun Exp $
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
  TMatrixD fR;  // (n x n) - upper triangular matrix
  TMatrixD fQ;  // (m x n) - orthogonal matrix
  TVectorD     fUp; // n - vector with Householder up's
  TVectorD     fW;  // n - vector with Householder beta's

  static Bool_t QRH(TMatrixD &q,TVectorD &diagR,TVectorD &up,TVectorD &w,Double_t tol);

public :

  TDecompQRH() {}
  TDecompQRH(const TMatrixD &m,Double_t tol = 0.0); // be careful for slicing in operator=
  TDecompQRH(const TDecompQRH &another);
  virtual ~TDecompQRH() {}

  virtual const TMatrixDBase &GetDecompMatrix() const { return fR; }
  virtual const TMatrixD     &GetR           () const { return fR; }
  virtual const TVectorD     &GetW           () const { return fW; }

  virtual Int_t  Decompose (const TMatrixDBase &a);
  virtual Bool_t Solve     (TVectorD &b);
  virtual Bool_t Solve     (TMatrixDColumn &b);
  virtual Bool_t TransSolve(TVectorD &b);
  virtual Bool_t TransSolve(TMatrixDColumn &b);
  virtual void   Det       (Double_t &d1,Double_t &d2);

  TDecompQRH &operator= (const TDecompQRH &source);

  ClassDef(TDecompQRH,1) // Matrix Decompositition QRH
};

#endif
