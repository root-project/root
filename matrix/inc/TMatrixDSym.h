// @(#)root/matrix:$Name:  $:$Id: TMatrixDSym.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDSym
#define ROOT_TMatrixDSym

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSym                                                          //
//                                                                      //
// Implementation of a symmetric matrix in the linear algebra package   //
//                                                                      //
// Note that in this implementation both matrix element m[i][j] and     //
// m[j][i] are updated and stored in memory . However, when making the  //                             
// object persistent only the upper right triangle is stored .          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif 

class TMatrixD;
class TVectorD;

class TMatrixFSym;
class TMatrixDSym : public TMatrixDBase {

protected:

  Double_t *fElements;  //[fNelems] elements themselves

  virtual void Allocate  (Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0);

  // Elementary constructors
  void AtMultA(const TMatrixD    &a,Int_t constr=1);
  void AtMultA(const TMatrixDSym &a,Int_t constr=1);

  void AMultA (const TMatrixDSym &a,Int_t constr=1) { AtMultA(a,constr); }

public:

  TMatrixDSym() { fIsOwner = kTRUE; fElements = 0; Invalidate(); }
  explicit TMatrixDSym(Int_t nrows);
  TMatrixDSym(Int_t row_lwb,Int_t row_upb);
  TMatrixDSym(Int_t nrows,const Double_t *data,Option_t *option="");
  TMatrixDSym(Int_t row_lwb,Int_t row_upb,const Double_t *data,Option_t *option="");
  TMatrixDSym(const TMatrixDSym &another);
  TMatrixDSym(const TMatrixFSym &another);

  TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixDSym &prototype);
  TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixD    &prototype);
  TMatrixDSym(const TMatrixDSymLazy &lazy_constructor);

  virtual ~TMatrixDSym() { Clear(); Invalidate(); }

  virtual const Double_t *GetMatrixArray  () const;
  virtual       Double_t *GetMatrixArray  ();

  virtual void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNelems,fElements); }

  void        Adopt         (Int_t nrows,Double_t *data);
  void        Adopt         (Int_t row_lwb,Int_t row_upb,Double_t *data);
  TMatrixDSym GetSub        (Int_t row_lwb,Int_t row_upb,Option_t *option="S") const;
  void        SetSub        (Int_t row_lwb,const TMatrixDSym &source);

  virtual  Double_t Determinant() const;
  virtual  void     Determinant(Double_t &d1,Double_t &d2) const;

  TMatrixDSym  &Zero        ();
  TMatrixDSym  &Abs         ();
  TMatrixDSym  &Sqr         ();
  TMatrixDSym  &Sqrt        ();
  TMatrixDSym  &UnitMatrix  ();
  TMatrixDSym  &Transpose   (const TMatrixDSym &source);
  TMatrixDSym  &NormByDiag  (const TVectorD &v,Option_t *option="D");

  // Either access a_ij as a(i,j)
  inline const Double_t &operator()(Int_t rown,Int_t coln) const;
  inline       Double_t &operator()(Int_t rown,Int_t coln)
                                   { return (Double_t&)((*(const TMatrixDSym *)this)(rown,coln)); }

  TMatrixDSym &operator= (const TMatrixDSym     &source);
  TMatrixDSym &operator= (const TMatrixDSymLazy &source);
  TMatrixDSym &operator= (Double_t val);
  TMatrixDSym &operator-=(Double_t val);
  TMatrixDSym &operator+=(Double_t val);
  TMatrixDSym &operator*=(Double_t val);

  TMatrixDSym &operator+=(const TMatrixDSym &source);
  TMatrixDSym &operator-=(const TMatrixDSym &source);

  TMatrixDSym &Apply(const TElementActionD    &action);
  TMatrixDSym &Apply(const TElementPosActionD &action);

  friend Bool_t       operator== (const TMatrixDSym &m1,     const TMatrixDSym &m2);
  friend TMatrixDSym  operator+  (const TMatrixDSym &source1,const TMatrixDSym &source2);
  friend TMatrixDSym  operator-  (const TMatrixDSym &source1,const TMatrixDSym &source2);
  friend TMatrixDSym  operator*  (      Double_t     val,    const TMatrixDSym &source );
  friend TMatrixDSym  operator*  (const TMatrixDSym &source,       Double_t     val    )
                                 { return operator*(val,source); }

  friend TMatrixDSym &Add        (TMatrixDSym &target,      Double_t     scalar,const TMatrixDSym &source);
  friend TMatrixDSym &ElementMult(TMatrixDSym &target,const TMatrixDSym &source);
  friend TMatrixDSym &ElementDiv (TMatrixDSym &target,const TMatrixDSym &source);

  ClassDef(TMatrixDSym,1) // Symmetric Matrix class (double precision)
};

inline const Double_t *TMatrixDSym::GetMatrixArray  () const { return fElements; }
inline       Double_t *TMatrixDSym::GetMatrixArray  ()       { return fElements; }
inline const Double_t &TMatrixDSym::operator()(Int_t rown,Int_t coln) const {
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  if (!(arown < fNrows && arown >= 0)) {
    printf("fRowLwb = %d\n",fRowLwb);
    printf("fNrows  = %d\n",fNrows);
    printf("arown   = %d\n",arown);
    printf("acoln   = %d\n",acoln);
  }
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

Bool_t       operator== (const TMatrixDSym &m1,     const TMatrixDSym  &m2);
TMatrixDSym  operator+  (const TMatrixDSym &source1,const TMatrixDSym  &source2);
TMatrixDSym  operator-  (const TMatrixDSym &source1,const TMatrixDSym  &source2);
TMatrixDSym  operator*  (      Double_t     val,    const TMatrixDSym  &source );
TMatrixDSym  operator*  (const TMatrixDSym &source,       Double_t      val    );

TMatrixDSym &Add        (TMatrixDSym &target,      Double_t     scalar,const TMatrixDSym &source);
TMatrixDSym &ElementMult(TMatrixDSym &target,const TMatrixDSym &source);
TMatrixDSym &ElementDiv (TMatrixDSym &target,const TMatrixDSym &source);

#endif
