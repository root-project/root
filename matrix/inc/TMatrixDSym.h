// @(#)root/matrix:$Name:  $:$Id: TMatrixDSym.h,v 1.13 2004/05/18 14:21:09 brun Exp $
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

  Double_t *fElements;  //![fNelems] elements themselves

  virtual void Allocate  (Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0,
                          Int_t nr_nonzeros = -1);

  // Elementary constructors
  void AtMultA(const TMatrixD    &a,Int_t constr=1);
  void AtMultA(const TMatrixDSym &a,Int_t constr=1);

  void AMultA (const TMatrixDSym &a,Int_t constr=1) { AtMultA(a,constr); }

public:

  TMatrixDSym() { fElements = 0; }
  explicit TMatrixDSym(Int_t nrows);
  TMatrixDSym(Int_t row_lwb,Int_t row_upb);
  TMatrixDSym(Int_t nrows,const Double_t *data,Option_t *option="");
  TMatrixDSym(Int_t row_lwb,Int_t row_upb,const Double_t *data,Option_t *option="");
  TMatrixDSym(const TMatrixDSym &another);
  TMatrixDSym(const TMatrixFSym &another);

  TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixDSym &prototype);
  TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixD    &prototype);
  TMatrixDSym(const TMatrixDSymLazy &lazy_constructor);

  virtual ~TMatrixDSym() { Clear(); }

  virtual const Double_t *GetMatrixArray  () const;
  virtual       Double_t *GetMatrixArray  ();
  virtual const Int_t    *GetRowIndexArray() const { return 0; }
  virtual       Int_t    *GetRowIndexArray()       { return 0; }
  virtual const Int_t    *GetColIndexArray() const { return 0; }
  virtual       Int_t    *GetColIndexArray()       { return 0; }

  virtual       TMatrixDBase &SetRowIndexArray(Int_t * /*data*/) { MayNotUse("SetRowIndexArray(Int_t *)"); return *this; }
  virtual       TMatrixDBase &SetColIndexArray(Int_t * /*data*/) { MayNotUse("SetColIndexArray(Int_t *)"); return *this; }

  virtual void   Clear      (Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNelems,fElements);
                                                          else fElements = 0; fNelems = 0; }
  virtual Bool_t IsSymmetric() const { return kTRUE; }


          TMatrixDSym  &Use           (Int_t nrows,Double_t *data);
          TMatrixDSym  &Use           (Int_t row_lwb,Int_t row_upb,Double_t *data);
          TMatrixDSym  &Use           (TMatrixDSym &a);

          TMatrixDSym  &GetSub        (Int_t row_lwb,Int_t row_upb,TMatrixDSym &target,Option_t *option="S") const;
  virtual TMatrixDBase &GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                       TMatrixDBase &target,Option_t *option="S") const;
          TMatrixDSym   GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
          TMatrixDSym  &SetSub        (Int_t row_lwb,const TMatrixDBase &source);
  virtual TMatrixDBase &SetSub        (Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source);

  virtual TMatrixDBase &SetMatrixArray(const Double_t *data, Option_t *option="");

  virtual TMatrixDBase &Shift         (Int_t row_shift,Int_t col_shift);
  virtual TMatrixDBase &ResizeTo      (Int_t nrows,Int_t ncols,Int_t nr_nonzeros=-1);
  virtual TMatrixDBase &ResizeTo      (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros=-1);
  inline  TMatrixDBase &ResizeTo      (const TMatrixDSym &m) {
                                        return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb()); }

  virtual Double_t Determinant   () const;
  virtual void     Determinant   (Double_t &d1,Double_t &d2) const;

          TMatrixDSym &Transpose (const TMatrixDSym &source);
  inline  TMatrixDSym &T         () { return this->Transpose(*this); }

  // Either access a_ij as a(i,j)
  inline       Double_t           operator()(Int_t rown,Int_t coln) const;
  inline       Double_t          &operator()(Int_t rown,Int_t coln);

  // or as a[i][j]
  inline const TMatrixDRow_const  operator[](Int_t rown) const { return TMatrixDRow_const(*this,rown); }
  inline       TMatrixDRow        operator[](Int_t rown)       { return TMatrixDRow      (*this,rown); }

  TMatrixDSym &operator= (const TMatrixDSym     &source);
  TMatrixDSym &operator= (const TMatrixFSym     &source);
  TMatrixDSym &operator= (const TMatrixDSymLazy &source);
  TMatrixDSym &operator= (Double_t val);
  TMatrixDSym &operator-=(Double_t val);
  TMatrixDSym &operator+=(Double_t val);
  TMatrixDSym &operator*=(Double_t val);

  TMatrixDSym &operator+=(const TMatrixDSym &source);
  TMatrixDSym &operator-=(const TMatrixDSym &source);

  TMatrixDBase &Apply(const TElementActionD    &action);
  TMatrixDBase &Apply(const TElementPosActionD &action);

  virtual TMatrixDBase &Randomize  (Double_t alpha,Double_t beta,Double_t &seed);
  virtual TMatrixDSym  &RandomizePD(Double_t alpha,Double_t beta,Double_t &seed);

  const TMatrixD EigenVectors(TVectorD &eigenValues) const;

  ClassDef(TMatrixDSym,1) // Symmetric Matrix class (double precision)
};

inline const Double_t    *TMatrixDSym::GetMatrixArray() const { return fElements; }
inline       Double_t    *TMatrixDSym::GetMatrixArray()       { return fElements; }
inline       TMatrixDSym &TMatrixDSym::Use           (Int_t nrows,Double_t *data) { return Use(0,nrows-1,data); }
inline       TMatrixDSym &TMatrixDSym::Use           (TMatrixDSym &a)
                                                              { return Use(a.GetRowLwb(),a.GetRowUpb(),a.GetMatrixArray()); }
inline       TMatrixDSym  TMatrixDSym::GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                      Option_t *option) const
                                                              {
                                                                TMatrixDSym tmp;
                                                                this->GetSub(row_lwb,row_upb,col_lwb,col_upb,tmp,option);
                                                                return tmp;
                                                              }

inline Double_t TMatrixDSym::operator()(Int_t rown,Int_t coln) const
{
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

inline Double_t &TMatrixDSym::operator()(Int_t rown,Int_t coln)
{
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
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
