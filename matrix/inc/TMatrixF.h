// @(#)root/matrix:$Name:  $:$Id: TMatrixF.h,v 1.6 2004/03/21 10:52:27 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixF
#define ROOT_TMatrixF

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixF                                                             //
//                                                                      //
// Implementation of a general matrix in the linear algebra package     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixFBase
#include "TMatrixFBase.h"
#endif

#ifdef CBLAS
#include <vecLib/vBLAS.h>
//#include <cblas.h>
#endif

class TMatrixD;
class TMatrixF : public TMatrixFBase {

protected:

  Float_t *fElements;  //[fNelems] elements themselves

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,
                        Int_t init = 0,Int_t nr_nonzero = -1);

  // Elementary constructors
  void AMultB (const TMatrixF     &a,const TMatrixF    &b,Int_t constr=1);
  void AMultB (const TMatrixF     &a,const TMatrixFSym &b,Int_t constr=1);
  void AMultB (const TMatrixFSym  &a,const TMatrixF    &b,Int_t constr=1);
  void AMultB (const TMatrixFSym  &a,const TMatrixFSym &b,Int_t constr=1);

  void AtMultB(const TMatrixF     &a,const TMatrixF    &b,Int_t constr=1);
  void AtMultB(const TMatrixF     &a,const TMatrixFSym &b,Int_t constr=1);
  void AtMultB(const TMatrixFSym  &a,const TMatrixF    &b,Int_t constr=1) { AMultB(a,b,constr); }
  void AtMultB(const TMatrixFSym  &a,const TMatrixFSym &b,Int_t constr=1) { AMultB(a,b,constr); }

public:

  TMatrixF() { fElements = 0; }
  TMatrixF(Int_t nrows,Int_t ncols);
  TMatrixF(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixF(Int_t nrows,Int_t ncols,const Float_t *data,Option_t *option="");
  TMatrixF(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Float_t *data,Option_t *option="");
  TMatrixF(const TMatrixF &another);
  TMatrixF(const TMatrixD &another);
  TMatrixF(const TMatrixFSym  &another);

  TMatrixF(EMatrixCreatorsOp1 op,const TMatrixF &prototype);
  TMatrixF(const TMatrixF     &a,EMatrixCreatorsOp2 op,const TMatrixF &b);
  TMatrixF(const TMatrixF     &a,EMatrixCreatorsOp2 op,const TMatrixFSym  &b);
  TMatrixF(const TMatrixFSym  &a,EMatrixCreatorsOp2 op,const TMatrixF &b);
  TMatrixF(const TMatrixFSym  &a,EMatrixCreatorsOp2 op,const TMatrixFSym  &b);
  TMatrixF(const TMatrixFLazy &lazy_constructor);

  virtual ~TMatrixF() { Clear(); }

  virtual const Float_t *GetMatrixArray  () const;
  virtual       Float_t *GetMatrixArray  ();

  virtual void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNelems,fElements); fNelems = 0; }

  void      Use         (Int_t nrows,Int_t ncols,Float_t *data);
  void      Use         (Int_t row_lwb,Int_t row_upb,
                         Int_t col_lwb,Int_t col_upb,Float_t *data);
  void      Use         (TMatrixF &a);
  TMatrixF  GetSub      (Int_t row_lwb,Int_t row_upb,
                         Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
  void      SetSub      (Int_t row_lwb,Int_t col_lwb,const TMatrixFBase &source);

  virtual Double_t Determinant() const;
  virtual void     Determinant(Double_t &d1,Double_t &d2) const;

  TMatrixF &Zero        ();
  TMatrixF &Abs         ();
  TMatrixF &Sqr         ();
  TMatrixF &Sqrt        ();
  TMatrixF &UnitMatrix  ();
  TMatrixF &Invert      (Double_t *det=0);
  TMatrixF &InvertFast  (Double_t *det=0);
  TMatrixF &Transpose   (const TMatrixF &source);

  inline TMatrixF &T    () { return this->Transpose(*this); }

  TMatrixF &NormByDiag  (const TVectorF &v,Option_t *option="D");
  TMatrixF &NormByColumn(const TVectorF &v,Option_t *option="D");
  TMatrixF &NormByRow   (const TVectorF &v,Option_t *option="D");

  inline void Mult(const TMatrixF    &a,const TMatrixF    &b) { AMultB(a,b,0); }
  inline void Mult(const TMatrixF    &a,const TMatrixFSym &b) { AMultB(a,b,0); }
  inline void Mult(const TMatrixFSym &a,const TMatrixF    &b) { AMultB(a,b,0); }

  // Either access a_ij as a(i,j)
  inline const Float_t           &operator()(Int_t rown,Int_t coln) const;
  inline       Float_t           &operator()(Int_t rown,Int_t coln)
                                             { return (Float_t&)((*(const TMatrixF *)this)(rown,coln)); }
  // or as a[i][j]
  inline const TMatrixFRow_const  operator[](Int_t rown) const { return TMatrixFRow_const(*this,rown); }
  inline       TMatrixFRow        operator[](Int_t rown)       { return TMatrixFRow      (*this,rown); }

  TMatrixF &operator= (const TMatrixF     &source);
  TMatrixF &operator= (const TMatrixD     &source);
  TMatrixF &operator= (const TMatrixFSym  &source);
  TMatrixF &operator= (const TMatrixFLazy &source);
  TMatrixF &operator= (Float_t val);
  TMatrixF &operator-=(Float_t val);
  TMatrixF &operator+=(Float_t val);
  TMatrixF &operator*=(Float_t val);

  TMatrixF &operator+=(const TMatrixF    &source);
  TMatrixF &operator+=(const TMatrixFSym &source);
  TMatrixF &operator-=(const TMatrixF    &source);
  TMatrixF &operator-=(const TMatrixFSym &source);

  TMatrixF &operator*=(const TMatrixF             &source);
  TMatrixF &operator*=(const TMatrixFSym          &source);
  TMatrixF &operator*=(const TMatrixFDiag_const   &diag);
  TMatrixF &operator/=(const TMatrixFDiag_const   &diag);
  TMatrixF &operator*=(const TMatrixFRow_const    &row);
  TMatrixF &operator/=(const TMatrixFRow_const    &row);
  TMatrixF &operator*=(const TMatrixFColumn_const &col);
  TMatrixF &operator/=(const TMatrixFColumn_const &col);

  TMatrixF &Apply(const TElementActionF    &action);
  TMatrixF &Apply(const TElementPosActionF &action);

  const TMatrixF EigenVectors(TVectorF &eigenValues) const;

  friend Bool_t    operator== (const TMatrixF    &m1,const TMatrixF    &m2);

  friend TMatrixF  operator+  (const TMatrixF    &source1,const TMatrixF    &source2);
  friend TMatrixF  operator+  (const TMatrixF    &source1,const TMatrixFSym &source2);
  friend TMatrixF  operator+  (const TMatrixFSym &source1,const TMatrixF    &source2);
  friend TMatrixF  operator-  (const TMatrixF    &source1,const TMatrixF    &source2);
  friend TMatrixF  operator-  (const TMatrixF    &source1,const TMatrixFSym &source2);
  friend TMatrixF  operator-  (const TMatrixFSym &source1,const TMatrixF    &source2);
  friend TMatrixF  operator*  (      Float_t      val,    const TMatrixF    &source );
  friend TMatrixF  operator*  (const TMatrixF     &source ,      Float_t      val   )
                              { return operator*(val,source); }
  friend TMatrixF  operator*  (const TMatrixF    &source1,const TMatrixF    &source2);
  friend TMatrixF  operator*  (const TMatrixF    &source1,const TMatrixFSym &source2);
  friend TMatrixF  operator*  (const TMatrixFSym &source1,const TMatrixF    &source2);
  friend TMatrixF  operator*  (const TMatrixFSym &source1,const TMatrixFSym &source2);

  friend TMatrixF &Add        (TMatrixF &target,      Float_t      scalar,const TMatrixF    &source);
  friend TMatrixF &Add        (TMatrixF &target,      Float_t      scalar,const TMatrixFSym &source);
  friend TMatrixF &ElementMult(TMatrixF &target,const TMatrixF    &source);
  friend TMatrixF &ElementMult(TMatrixF &target,const TMatrixFSym &source);
  friend TMatrixF &ElementDiv (TMatrixF &target,const TMatrixF    &source);
  friend TMatrixF &ElementDiv (TMatrixF &target,const TMatrixFSym &source);

  ClassDef(TMatrixF,3) // Matrix class (single precision)
};

class TMatrix : public TMatrixF {
public :
  TMatrix() {}
  TMatrix(Int_t nrows,Int_t ncols) : TMatrixF(nrows,ncols) {}
  TMatrix(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb) :
    TMatrixF(row_lwb,row_upb,col_lwb,col_upb) {}
  TMatrix(Int_t nrows,Int_t ncols,const Float_t *data,Option_t *option="") :
    TMatrixF(nrows,ncols,data,option) {}
  TMatrix(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Float_t *data,Option_t *option="") :
    TMatrixF(row_lwb,row_upb,col_lwb,col_upb,data,option) {}
  TMatrix(const TMatrixF     &another) : TMatrixF(another) {}
  TMatrix(const TMatrixD     &another) : TMatrixF(another) {}
  TMatrix(const TMatrixFSym  &another) : TMatrixF(another) {}

  TMatrix(EMatrixCreatorsOp1 op,const TMatrixF &prototype)                  : TMatrixF(op,prototype) {}
  TMatrix(const TMatrixF    &a,EMatrixCreatorsOp2 op,const TMatrixF &b)     : TMatrixF(a,op,b) {}
  TMatrix(const TMatrixF    &a,EMatrixCreatorsOp2 op,const TMatrixFSym  &b) : TMatrixF(a,op,b) {}
  TMatrix(const TMatrixFSym &a,EMatrixCreatorsOp2 op,const TMatrixF &b)     : TMatrixF(a,op,b) {}
  TMatrix(const TMatrixFSym &a,EMatrixCreatorsOp2 op,const TMatrixFSym  &b) : TMatrixF(a,op,b) {}
  TMatrix(const TMatrixFLazy &lazy_constructor)                             : TMatrixF(lazy_constructor) {}

  virtual ~TMatrix() {}
  ClassDef(TMatrix,3)  // Matrix class (single precision)
};

inline const Float_t  *TMatrixF::GetMatrixArray () const { return fElements; }
inline       Float_t  *TMatrixF::GetMatrixArray ()       { return fElements; }
inline       void      TMatrixF::Use(TMatrixF &a) { Use(a.GetRowLwb(),a.GetRowUpb(),a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray()); }
inline const Float_t  &TMatrixF::operator    ()(Int_t rown,Int_t coln) const {
  Assert(IsValid());
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  return (fElements[arown*fNcols+acoln]);
}

Bool_t    operator== (const TMatrixF    &m1,const TMatrixF    &m2);

TMatrixF  operator+  (const TMatrixF    &source1,const TMatrixF    &source2);
TMatrixF  operator+  (const TMatrixF    &source1,const TMatrixFSym &source2);
TMatrixF  operator+  (const TMatrixFSym &source1,const TMatrixF    &source2);
TMatrixF  operator-  (const TMatrixF    &source1,const TMatrixF    &source2);
TMatrixF  operator-  (const TMatrixF    &source1,const TMatrixFSym &source2);
TMatrixF  operator-  (const TMatrixFSym &source1,const TMatrixF    &source2);
TMatrixF  operator*  (      Float_t      val,    const TMatrixF    &source );
TMatrixF  operator*  (const TMatrixF    &source,       Float_t      val    );
TMatrixF  operator*  (const TMatrixF    &source1,const TMatrixF    &source2);
TMatrixF  operator*  (const TMatrixF    &source1,const TMatrixFSym &source2);
TMatrixF  operator*  (const TMatrixFSym &source1,const TMatrixF    &source2);
TMatrixF  operator*  (const TMatrixFSym &source1,const TMatrixFSym &source2);

TMatrixF &Add        (TMatrixF &target,       Float_t     scalar,const TMatrixF    &source);
TMatrixF &Add        (TMatrixF &target,       Float_t     scalar,const TMatrixFSym &source);
TMatrixF &ElementMult(TMatrixF &target,const TMatrixF    &source);
TMatrixF &ElementMult(TMatrixF &target,const TMatrixFSym &source);
TMatrixF &ElementDiv (TMatrixF &target,const TMatrixF    &source);
TMatrixF &ElementDiv (TMatrixF &target,const TMatrixFSym &source);

#endif
