// @(#)root/matrix:$Name:  $:$Id: TMatrixD.h,v 1.28 2004/01/26 11:28:07 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixD
#define ROOT_TMatrixD

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixD                                                             //
//                                                                      //
// Implementation of a general matrix in the linear algebra package     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif

#ifdef CBLAS
#include <vecLib/vBLAS.h>
//#include <cblas.h>
#endif

class TMatrixF;
class TMatrixD : public TMatrixDBase {

protected:

  Double_t *fElements;  //[fNelems] elements themselves

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0);

  // Elementary constructors
  void AMultB (const TMatrixD     &a,const TMatrixD    &b,Int_t constr=1);
  void AMultB (const TMatrixD     &a,const TMatrixDSym &b,Int_t constr=1);
  void AMultB (const TMatrixDSym  &a,const TMatrixD    &b,Int_t constr=1);
  void AMultB (const TMatrixDSym  &a,const TMatrixDSym &b,Int_t constr=1);

  void AtMultB(const TMatrixD     &a,const TMatrixD    &b,Int_t constr=1);
  void AtMultB(const TMatrixD     &a,const TMatrixDSym &b,Int_t constr=1);
  void AtMultB(const TMatrixDSym  &a,const TMatrixD    &b,Int_t constr=1) { AMultB(a,b,constr); }
  void AtMultB(const TMatrixDSym  &a,const TMatrixDSym &b,Int_t constr=1) { AMultB(a,b,constr); }

public:

  TMatrixD() { fIsOwner = kTRUE; fElements = 0; Invalidate(); }
  TMatrixD(Int_t nrows,Int_t ncols);
  TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixD(Int_t nrows,Int_t ncols,const Double_t *data,Option_t *option="");
  TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Double_t *data,Option_t *option="");
  TMatrixD(const TMatrixD &another);
  TMatrixD(const TMatrixF &another);
  TMatrixD(const TMatrixDSym  &another);

  TMatrixD(EMatrixCreatorsOp1 op,const TMatrixD &prototype);
  TMatrixD(const TMatrixD     &a,EMatrixCreatorsOp2 op,const TMatrixD &b);
  TMatrixD(const TMatrixD     &a,EMatrixCreatorsOp2 op,const TMatrixDSym  &b);
  TMatrixD(const TMatrixDSym  &a,EMatrixCreatorsOp2 op,const TMatrixD &b);
  TMatrixD(const TMatrixDSym  &a,EMatrixCreatorsOp2 op,const TMatrixDSym  &b);
  TMatrixD(const TMatrixDLazy &lazy_constructor);

  virtual ~TMatrixD() { Clear(); Invalidate(); }

  virtual const Double_t *GetMatrixArray  () const;
  virtual       Double_t *GetMatrixArray  ();

  virtual void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNelems,fElements); }

  void      Adopt       (Int_t nrows,Int_t ncols,Double_t *data);
  void      Adopt       (Int_t row_lwb,Int_t row_upb,
                         Int_t col_lwb,Int_t col_upb,Double_t *data);
  void      Adopt       (TMatrixD &a);
  TMatrixD  GetSub      (Int_t row_lwb,Int_t row_upb,
                         Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
  void      SetSub      (Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source);

  virtual Double_t Determinant() const;
  virtual void     Determinant(Double_t &d1,Double_t &d2) const;

  TMatrixD &Zero        ();
  TMatrixD &Abs         ();
  TMatrixD &Sqr         ();
  TMatrixD &Sqrt        ();
  TMatrixD &UnitMatrix  ();
  TMatrixD &Invert      (Double_t *det=0);
  TMatrixD &InvertFast  (Double_t *det=0);
  TMatrixD &Transpose   (const TMatrixD &source);

  inline TMatrixD &T    () { return this->Transpose(*this); }

  TMatrixD &NormByDiag  (const TVectorD &v,Option_t *option="D");
  TMatrixD &NormByColumn(const TVectorD &v,Option_t *option="D");
  TMatrixD &NormByRow   (const TVectorD &v,Option_t *option="D");

  inline void Mult(const TMatrixD    &a,const TMatrixD    &b) { AMultB(a,b,0); }
  inline void Mult(const TMatrixD    &a,const TMatrixDSym &b) { AMultB(a,b,0); }
  inline void Mult(const TMatrixDSym &a,const TMatrixD    &b) { AMultB(a,b,0); }

  // Either access a_ij as a(i,j)
  inline const Double_t &operator()(Int_t rown,Int_t coln) const;
  inline       Double_t &operator()(Int_t rown,Int_t coln)
                                    { return (Double_t&)((*(const TMatrixD *)this)(rown,coln)); }

  TMatrixD &operator= (const TMatrixD     &source);
  TMatrixD &operator= (const TMatrixF     &source);
  TMatrixD &operator= (const TMatrixDSym  &source);
  TMatrixD &operator= (const TMatrixDLazy &source);
  TMatrixD &operator= (Double_t val);
  TMatrixD &operator-=(Double_t val);
  TMatrixD &operator+=(Double_t val);
  TMatrixD &operator*=(Double_t val);

  TMatrixD &operator+=(const TMatrixD    &source);
  TMatrixD &operator+=(const TMatrixDSym &source);
  TMatrixD &operator-=(const TMatrixD    &source);
  TMatrixD &operator-=(const TMatrixDSym &source);

  TMatrixD &operator*=(const TMatrixD             &source);
  TMatrixD &operator*=(const TMatrixDSym          &source);
  TMatrixD &operator*=(const TMatrixDDiag_const   &diag);
  TMatrixD &operator/=(const TMatrixDDiag_const   &diag);
  TMatrixD &operator*=(const TMatrixDRow_const    &row);
  TMatrixD &operator/=(const TMatrixDRow_const    &row);
  TMatrixD &operator*=(const TMatrixDColumn_const &col);
  TMatrixD &operator/=(const TMatrixDColumn_const &col);

  TMatrixD &Apply(const TElementActionD    &action);
  TMatrixD &Apply(const TElementPosActionD &action);

  friend Bool_t    operator== (const TMatrixD    &m1,const TMatrixD    &m2);

  friend TMatrixD  operator+  (const TMatrixD    &source1,const TMatrixD    &source2);
  friend TMatrixD  operator+  (const TMatrixD    &source1,const TMatrixDSym &source2);
  friend TMatrixD  operator+  (const TMatrixDSym &source1,const TMatrixD    &source2);
  friend TMatrixD  operator-  (const TMatrixD    &source1,const TMatrixD    &source2);
  friend TMatrixD  operator-  (const TMatrixD    &source1,const TMatrixDSym &source2);
  friend TMatrixD  operator-  (const TMatrixDSym &source1,const TMatrixD    &source2);
  friend TMatrixD  operator*  (      Double_t     val,    const TMatrixD    &source );
  friend TMatrixD  operator*  (const TMatrixD     &source ,      Double_t     val   )
                              { return operator*(val,source); }
  friend TMatrixD  operator*  (const TMatrixD    &source1,const TMatrixD    &source2);
  friend TMatrixD  operator*  (const TMatrixD    &source1,const TMatrixDSym &source2);
  friend TMatrixD  operator*  (const TMatrixDSym &source1,const TMatrixD    &source2);
  friend TMatrixD  operator*  (const TMatrixDSym &source1,const TMatrixDSym &source2);

  friend TMatrixD &Add        (TMatrixD &target,      Double_t     scalar,const TMatrixD    &source);
  friend TMatrixD &Add        (TMatrixD &target,      Double_t     scalar,const TMatrixDSym &source);
  friend TMatrixD &ElementMult(TMatrixD &target,const TMatrixD    &source);
  friend TMatrixD &ElementMult(TMatrixD &target,const TMatrixDSym &source);
  friend TMatrixD &ElementDiv (TMatrixD &target,const TMatrixD    &source);
  friend TMatrixD &ElementDiv (TMatrixD &target,const TMatrixDSym &source);

  ClassDef(TMatrixD,3) // Matrix class (double precision)
};

inline const Double_t *TMatrixD::GetMatrixArray () const { return fElements; }
inline       Double_t *TMatrixD::GetMatrixArray ()       { return fElements; }
inline       void      TMatrixD::Adopt(TMatrixD &a) { Adopt(a.GetRowLwb(),a.GetRowUpb(),a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray()); }
inline const Double_t &TMatrixD::operator    ()(Int_t rown,Int_t coln) const {
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

Bool_t    operator== (const TMatrixD    &m1,const TMatrixD    &m2);
//Bool_t    operator== (const TMatrixDSym &m1,const TMatrixD    &m2);
//Bool_t    operator== (const TMatrixD    &m1,const TMatrixDSym &m2);

TMatrixD  operator+  (const TMatrixD    &source1,const TMatrixD    &source2);
TMatrixD  operator+  (const TMatrixD    &source1,const TMatrixDSym &source2);
TMatrixD  operator+  (const TMatrixDSym &source1,const TMatrixD    &source2);
TMatrixD  operator-  (const TMatrixD    &source1,const TMatrixD    &source2);
TMatrixD  operator-  (const TMatrixD    &source1,const TMatrixDSym &source2);
TMatrixD  operator-  (const TMatrixDSym &source1,const TMatrixD    &source2);
TMatrixD  operator*  (      Double_t     val,    const TMatrixD    &source );
TMatrixD  operator*  (const TMatrixD    &source,       Double_t     val    );
TMatrixD  operator*  (const TMatrixD    &source1,const TMatrixD    &source2);
TMatrixD  operator*  (const TMatrixD    &source1,const TMatrixDSym &source2);
TMatrixD  operator*  (const TMatrixDSym &source1,const TMatrixD    &source2);
TMatrixD  operator*  (const TMatrixDSym &source1,const TMatrixDSym &source2);

TMatrixD &Add        (TMatrixD &target,       Double_t    scalar,const TMatrixD    &source);
TMatrixD &Add        (TMatrixD &target,       Double_t    scalar,const TMatrixDSym &source);
TMatrixD &ElementMult(TMatrixD &target,const TMatrixD    &source);
TMatrixD &ElementMult(TMatrixD &target,const TMatrixDSym &source);
TMatrixD &ElementDiv (TMatrixD &target,const TMatrixD    &source);
TMatrixD &ElementDiv (TMatrixD &target,const TMatrixDSym &source);

#endif
