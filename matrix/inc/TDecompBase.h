// @(#)root/matrix:$Name:  $:$Id: TDecompBase.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompBase
#define ROOT_TDecompBase

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Decomposition Base class                                              //
//                                                                       //
// This class forms the base for all the decompositions methods in the   //
// linear algebra package .                                              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

//#include <limits>

#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif

class TDecompBase : public TObject
{
protected :
  Int_t    fStatus;    // decomposition status
  Double_t fTol;       // sqrt(epsilon); epsilon is smallest number number so that  1+epsilon > 1
  Double_t fDet1;      // determinant mantissa
  Double_t fDet2;      // determinant exponent for powers of 2
  Double_t fCondition; // matrix condition number

  Int_t Hager(Double_t& est,Int_t iter=5);

  virtual const TMatrixDBase &GetDecompMatrix() const = 0;

public :
  enum EMatrixDecompStat { kInit=0,kDecomposed=1,kDetermined=2,kCondition=4,kSingular=8 };
  enum {kWorkMax = 100}; // size of work array's in several routines

  TDecompBase();
  TDecompBase(const TDecompBase &another);
  virtual ~TDecompBase() {};

          inline       Int_t         GetStatus      () const { return fStatus; }
          inline       Double_t      GetTol         () const { return fTol; }
          inline       Double_t      GetDet1        () const { return fDet1; }
          inline       Double_t      GetDet2        () const { return fDet2; }
          inline       Double_t      GetCondition   () const { return fCondition; }
  virtual              Int_t         GetNrows       () const = 0;
  virtual              Int_t         GetNcols       () const = 0;
          inline       Double_t      SetTol         (Double_t tol);

  virtual Double_t Condition ();
  virtual void     Det       (Double_t &d1,Double_t &d2);
  virtual Int_t    Decompose (const TMatrixDBase &a) = 0;
  virtual Bool_t   Solve     (TVectorD& b)           = 0;
  virtual Bool_t   TransSolve(TVectorD& b)           = 0;
  virtual Bool_t   Solve     (TMatrixDColumn& b)     = 0;
  virtual Bool_t   TransSolve(TMatrixDColumn& b)     = 0;

  virtual Bool_t   MultiSolve(TMatrixDBase& B);
  virtual void     Invert    (TMatrixDBase& inv);

  static  void     DiagProd  (const TVectorD &diag,Double_t tol,Double_t &d1,Double_t &d2);

  TDecompBase &operator= (const TDecompBase &source);

  ClassDef(TDecompBase,1) // Matrix Decompositition Base
};

Double_t TDecompBase::SetTol(Double_t newTol) 
{
  const Double_t oldTol = fTol; 
  if (newTol >= 0.0) 
    fTol = newTol; 
  return oldTol; 
}

Bool_t DefHouseHolder  (const TVectorD &vc,Int_t    lp,Int_t    l,Double_t &up,Double_t &b,Double_t tol=0.0);
void   ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t b,Int_t     lp,Int_t     l,TMatrixDRow &cr);
void   ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t b,Int_t     lp,Int_t     l,TMatrixDColumn &cc);
void   ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t b,Int_t     lp,Int_t     l,TVectorD &cv);
void   DefGivens       (Double_t  v1,Double_t  v2,Double_t &c,Double_t &s);
void   DefAplGivens    (Double_t &v1,Double_t &v2,Double_t &c,Double_t &s);
void   ApplyGivens     (Double_t &z1,Double_t &z2,Double_t  c,Double_t  s);

#endif
