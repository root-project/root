// @(#)root/matrix:$Name:  $:$Id: TMatrixFSym.h,v 1.10 2004/05/12 18:24:58 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFSym
#define ROOT_TMatrixFSym

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFSym                                                          //
//                                                                      //
// Implementation of a symmetric matrix in the linear algebra package   //
//                                                                      //
// Note that in this implementation both matrix element m[i][j] and     //
// m[j][i] are updated and stored in memory . However, when making the  //
// object persistent only the upper right triangle is stored .          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixFBase
#include "TMatrixFBase.h"
#endif

class TMatrixF;
class TVectorF;

class TMatrixFSym : public TMatrixFBase {

protected:

  Float_t *fElements;  //![fNelems] elements themselves

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0,
                        Int_t nr_nonzeros = -1);

  // Elementary constructors
  void AtMultA(const TMatrixF    &a,Int_t constr=1);
  void AtMultA(const TMatrixFSym &a,Int_t constr=1);

  void AMultA (const TMatrixFSym &a,Int_t constr=1) { AtMultA(a,constr); }

public:

  TMatrixFSym() { fElements = 0; }
  explicit TMatrixFSym(Int_t nrows);
  TMatrixFSym(Int_t row_lwb,Int_t row_upb);
  TMatrixFSym(Int_t nrows,const Float_t *data,Option_t *option="");
  TMatrixFSym(Int_t row_lwb,Int_t row_upb,const Float_t *data,Option_t *option="");
  TMatrixFSym(const TMatrixFSym &another);

  TMatrixFSym(EMatrixCreatorsOp1 op,const TMatrixFSym &prototype);
  TMatrixFSym(EMatrixCreatorsOp1 op,const TMatrixF    &prototype);
  TMatrixFSym(const TMatrixFSymLazy &lazy_constructor);

  virtual ~TMatrixFSym() { Clear(); }

  virtual const Float_t *GetMatrixArray  () const;
  virtual       Float_t *GetMatrixArray  ();
  virtual const Int_t   *GetRowIndexArray() const { return 0; }
  virtual       Int_t   *GetRowIndexArray()       { return 0; }
  virtual const Int_t   *GetColIndexArray() const { return 0; }
  virtual       Int_t   *GetColIndexArray()       { return 0; }
  virtual       void     SetRowIndexArray(Int_t */*data*/) { MayNotUse("SetRowIndexArray(Int_t *)"); }
  virtual       void     SetColIndexArray(Int_t */*data*/) { MayNotUse("SetColIndexArray(Int_t *)"); }

  virtual void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNelems,fElements); fNelems = 0; }

  void         Use           (Int_t row_lwb,Int_t row_upb,Float_t *data);
  void         Use           (Int_t nrows,Float_t *data);
  void         Use           (TMatrixFSym &a);
  TMatrixFSym  GetSub        (Int_t row_lwb,Int_t row_upb,Option_t *option="S") const;
  void         SetSub        (Int_t row_lwb,const TMatrixFBase &source);
  void         SetSub        (Int_t row_lwb,Int_t col_lwb,const TMatrixFBase &source);

  virtual void SetMatrixArray(const Float_t *data, Option_t *option="");

  virtual void Shift         (Int_t row_shift,Int_t col_shift);
  virtual void ResizeTo      (Int_t nrows,Int_t ncols,Int_t nr_nonzeros=-1);
  virtual void ResizeTo      (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros=-1);
  inline  void ResizeTo      (const TMatrixFSym &m) {
    ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
  }

  virtual Double_t Determinant   () const;
  virtual void     Determinant   (Double_t &d1,Double_t &d2) const;

          TMatrixFSym &Transpose (const TMatrixFSym &source);
  inline  TMatrixFSym &T         () { return this->Transpose(*this); }

  // Either access a_ij as a(i,j)
  inline Float_t            operator()(Int_t rown,Int_t coln) const;
  inline Float_t           &operator()(Int_t rown,Int_t coln);

  // or as a[i][j]
  inline const TMatrixFRow_const  operator[](Int_t rown) const { return TMatrixFRow_const(*this,rown); }
  inline       TMatrixFRow        operator[](Int_t rown)       { return TMatrixFRow      (*this,rown); }

  TMatrixFSym &operator= (const TMatrixFSym     &source);
  TMatrixFSym &operator= (const TMatrixFSymLazy &source);
  TMatrixFSym &operator= (Float_t val);
  TMatrixFSym &operator-=(Float_t val);
  TMatrixFSym &operator+=(Float_t val);
  TMatrixFSym &operator*=(Float_t val);

  TMatrixFSym &operator+=(const TMatrixFSym &source);
  TMatrixFSym &operator-=(const TMatrixFSym &source);

  TMatrixFBase &Apply(const TElementActionF    &action);
  TMatrixFBase &Apply(const TElementPosActionF &action);

  virtual void Randomize  (Float_t alpha,Float_t beta,Double_t &seed);
  virtual void RandomizePD(Float_t alpha,Float_t beta,Double_t &seed);

  const TMatrixF EigenVectors(TVectorF &eigenValues) const;

  ClassDef(TMatrixFSym,1) // Symmetric Matrix class (single precision)
};

inline const Float_t  *TMatrixFSym::GetMatrixArray() const { return fElements; }
inline       Float_t  *TMatrixFSym::GetMatrixArray()       { return fElements; }
inline       void      TMatrixFSym::Use           (Int_t nrows,Float_t *data) { Use(0,nrows-1,data); }
inline       void      TMatrixFSym::Use           (TMatrixFSym &a) { Use(a.GetRowLwb(),a.GetRowUpb(),a.GetMatrixArray()); }

inline Float_t TMatrixFSym::operator()(Int_t rown,Int_t coln) const {
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

inline Float_t &TMatrixFSym::operator()(Int_t rown,Int_t coln) {
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

Bool_t       operator== (const TMatrixFSym &m1,     const TMatrixFSym  &m2);
TMatrixFSym  operator+  (const TMatrixFSym &source1,const TMatrixFSym  &source2);
TMatrixFSym  operator-  (const TMatrixFSym &source1,const TMatrixFSym  &source2);
TMatrixFSym  operator*  (      Float_t      val,    const TMatrixFSym  &source );
TMatrixFSym  operator*  (const TMatrixFSym &source,       Float_t       val    );

TMatrixFSym &Add        (TMatrixFSym &target,      Float_t      scalar,const TMatrixFSym &source);
TMatrixFSym &ElementMult(TMatrixFSym &target,const TMatrixFSym &source);
TMatrixFSym &ElementDiv (TMatrixFSym &target,const TMatrixFSym &source);

#endif
