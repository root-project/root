// @(#)root/matrix:$Name:  $:$Id: TMatrixDSparse.h,v 1.5 2004/05/18 14:01:04 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Feb 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDSparse
#define ROOT_TMatrixDSparse

#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif

#ifdef CBLAS
#include <vecLib/vBLAS.h>
//#include <cblas.h>
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSparse                                                       //
//                                                                      //
// Implementation of a general sparse matrix in the Harwell-Boeing      //
// format                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDSparse : public TMatrixDBase {

protected:

  Int_t    *fRowIndex;  //[fNrowIndex] row index
  Int_t    *fColIndex;  //[fNelems]    column index
  Double_t *fElements;  //[fNelems]

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,
                        Int_t init = 0,Int_t nr_nonzeros = 0);

          void SetSparseIndexAB(const TMatrixDSparse &a,const TMatrixDSparse &b);

  // Elementary constructors
  void AMultB (const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr=1) {
               const TMatrixDSparse bt(TMatrixDSparse::kTransposed,b); AMultBt(a,bt,constr); }
  void AMultB (const TMatrixDSparse &a,const TMatrixD       &b,Int_t constr=1) {
               const TMatrixDSparse bsp = b;
			   const TMatrixDSparse bt(TMatrixDSparse::kTransposed,bsp); AMultBt(a,bt,constr); }
  void AMultB (const TMatrixD       &a,const TMatrixDSparse &b,Int_t constr=1) {
               const TMatrixDSparse bt(TMatrixDSparse::kTransposed,b); AMultBt(a,bt,constr); }

  void AMultBt(const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr=1);
  void AMultBt(const TMatrixDSparse &a,const TMatrixD       &b,Int_t constr=1);
  void AMultBt(const TMatrixD       &a,const TMatrixDSparse &b,Int_t constr=1);

  void APlusB (const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr=1);
  void APlusB (const TMatrixDSparse &a,const TMatrixD       &b,Int_t constr=1);
  void APlusB (const TMatrixD       &a,const TMatrixDSparse &b,Int_t constr=1) { APlusB(b,a,constr); }

  void AMinusB(const TMatrixDSparse &a,const TMatrixDSparse &b,Int_t constr=1);
  void AMinusB(const TMatrixDSparse &a,const TMatrixD       &b,Int_t constr=1);
  void AMinusB(const TMatrixD       &a,const TMatrixDSparse &b,Int_t constr=1);

public:

  TMatrixDSparse() { fElements = 0; fRowIndex = 0; fColIndex = 0; }
  TMatrixDSparse(Int_t nrows,Int_t ncols);
  TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixDSparse(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros,
                 Int_t *row, Int_t *col,Double_t *data);
  TMatrixDSparse(const TMatrixDSparse &another);
  TMatrixDSparse(const TMatrixD       &another);

  TMatrixDSparse(EMatrixCreatorsOp1 op,const TMatrixDSparse &prototype);
  TMatrixDSparse(const TMatrixDSparse &a,EMatrixCreatorsOp2 op,const TMatrixDSparse &b);

  virtual ~TMatrixDSparse() { Clear(); }

  virtual const Double_t *GetMatrixArray  () const;
  virtual       Double_t *GetMatrixArray  ();
  virtual const Int_t    *GetRowIndexArray() const;
  virtual       Int_t    *GetRowIndexArray();
  virtual const Int_t    *GetColIndexArray() const;
  virtual       Int_t    *GetColIndexArray();

  virtual TMatrixDBase   &SetRowIndexArray(Int_t *data) { memmove(fRowIndex,data,(fNrows+1)*sizeof(Int_t)); return *this; }
  virtual TMatrixDBase   &SetColIndexArray(Int_t *data) { memmove(fColIndex,data,fNelems*sizeof(Int_t)); return *this; }

  virtual void            GetMatrix2Array(Double_t *data,Option_t *option="") const;
  virtual TMatrixDBase   &SetMatrixArray (const Double_t *data,Option_t * /*option*/="")
                                          { memcpy(fElements,data,fNelems*sizeof(Double_t)); return *this; }
  virtual TMatrixDBase   &SetMatrixArray (Int_t nr_nonzeros,Int_t *irow,Int_t *icol,Double_t *data);
          TMatrixDSparse &SetSparseIndex (const TMatrixDBase &another);
          TMatrixDSparse &SetSparseIndex (Int_t nelem_new);
  virtual TMatrixDBase   &InsertRow      (Int_t row,Int_t col,const Double_t *v,Int_t n=-1);
  virtual void            ExtractRow     (Int_t row,Int_t col,      Double_t *v,Int_t n=-1) const;

  virtual TMatrixDBase   &ResizeTo(Int_t nrows,Int_t ncols,Int_t nr_nonzeros=-1);
  virtual TMatrixDBase   &ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros=-1);
  inline  TMatrixDBase   &ResizeTo(const TMatrixDSparse &m) {return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),
                                                                             m.GetColUpb(),m.GetNoElements()); }

  virtual void Clear(Option_t * /*option*/ ="") { if (fIsOwner) {
                                                    if (fElements) delete [] fElements; fElements = 0;
                                                    if (fRowIndex) delete [] fRowIndex; fRowIndex = 0;
                                                    if (fColIndex) delete [] fColIndex; fColIndex = 0;
                                                  }
                                                  fNelems    = 0;
                                                  fNrowIndex = 0;
                                                }

          TMatrixDSparse &Use   (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros,
                                 Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData);
          TMatrixDSparse &Use   (Int_t nrows,Int_t ncols,Int_t nr_nonzeros,
                                 Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData);
          TMatrixDSparse &Use   (TMatrixDSparse &a);
  virtual TMatrixDBase   &GetSub(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                  TMatrixDBase &target,Option_t *option="S") const;
          TMatrixDSparse  GetSub(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
  virtual TMatrixDBase   &SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source);

  virtual Bool_t IsSymmetric() const { return (*this == TMatrixDSparse(TMatrixDSparse::kTransposed,*this)); }
  TMatrixDSparse &Transpose (const TMatrixDSparse &source);
  inline TMatrixDSparse &T () { return this->Transpose(*this); }

  inline void Mult(const TMatrixDSparse &a,const TMatrixDSparse &b) { AMultB(a,b,0); }

  virtual TMatrixDBase &Zero       ();
  virtual TMatrixDBase &UnitMatrix ();

  virtual Double_t RowNorm () const;
  virtual Double_t ColNorm () const;
  virtual Int_t    NonZeros() const { return fNelems; }

  virtual TMatrixDBase &NormByDiag(const TVectorD &/*v*/,Option_t * /*option*/)
                                    { MayNotUse("NormByDiag"); return *this; }

  // Either access a_ij as a(i,j)
  inline       Double_t                 operator()(Int_t rown,Int_t coln) const;
               Double_t                &operator()(Int_t rown,Int_t coln);

  // or as a[i][j]
  inline const TMatrixDSparseRow_const  operator[](Int_t rown) const { return TMatrixDSparseRow_const(*this,rown); }
  inline       TMatrixDSparseRow        operator[](Int_t rown)       { return TMatrixDSparseRow      (*this,rown); }

  TMatrixDSparse &operator=(const TMatrixD       &source);
  TMatrixDSparse &operator=(const TMatrixDSparse &source);

  TMatrixDSparse &operator= (Double_t val);
  TMatrixDSparse &operator-=(Double_t val);
  TMatrixDSparse &operator+=(Double_t val);
  TMatrixDSparse &operator*=(Double_t val);

  TMatrixDSparse &operator+=(const TMatrixDSparse &source) { TMatrixDSparse tmp(*this);
                                                             if (this == &source) APlusB (tmp,tmp);
                                                             else                 APlusB (tmp,source); return *this; }
  TMatrixDSparse &operator+=(const TMatrixD       &source) { TMatrixDSparse tmp(*this); APlusB(tmp,source); return *this; }
  TMatrixDSparse &operator-=(const TMatrixDSparse &source) { TMatrixDSparse tmp(*this);
                                                             if (this == &source) AMinusB (tmp,tmp);
                                                             else                 AMinusB(tmp,source); return *this; }
  TMatrixDSparse &operator-=(const TMatrixD       &source) { TMatrixDSparse tmp(*this); AMinusB(tmp,source); return *this; }
  TMatrixDSparse &operator*=(const TMatrixDSparse &source) { TMatrixDSparse tmp(*this);
                                                             if (this == &source) AMultB (tmp,tmp);
                                                             else                 AMultB (tmp,source); return *this; }
  TMatrixDSparse &operator*=(const TMatrixD       &source) { TMatrixDSparse tmp(*this); AMultB(tmp,source); return *this; }

  virtual TMatrixDBase   &Randomize  (Double_t alpha,Double_t beta,Double_t &seed);
  virtual TMatrixDSparse &RandomizePD(Double_t alpha,Double_t beta,Double_t &seed);

  ClassDef(TMatrixDSparse,2) // Sparse Matrix class (double precision)
};

inline const Double_t *TMatrixDSparse::GetMatrixArray  () const { return fElements; }
inline       Double_t *TMatrixDSparse::GetMatrixArray  ()       { return fElements; }
inline const Int_t    *TMatrixDSparse::GetRowIndexArray() const { return fRowIndex; }
inline       Int_t    *TMatrixDSparse::GetRowIndexArray()       { return fRowIndex; }
inline const Int_t    *TMatrixDSparse::GetColIndexArray() const { return fColIndex; }
inline       Int_t    *TMatrixDSparse::GetColIndexArray()       { return fColIndex; }

inline       TMatrixDSparse &TMatrixDSparse::Use   (Int_t nrows,Int_t ncols,Int_t nr_nonzeros,
                                                    Int_t *pRowIndex,Int_t *pColIndex,Double_t *pData)
                                                      { return Use(0,nrows-1,0,ncols-1,nr_nonzeros,pRowIndex,pColIndex,pData); }
inline       TMatrixDSparse &TMatrixDSparse::Use   (TMatrixDSparse &a)
                                                      { Assert(a.IsValid());
                                                         return Use(a.GetRowLwb(),a.GetRowUpb(),a.GetColLwb(),a.GetColUpb(),
                                                                    a.GetNoElements(),a.GetRowIndexArray(),
                                                                    a.GetColIndexArray(),a.GetMatrixArray()); }
inline       TMatrixDSparse  TMatrixDSparse::GetSub(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                    Option_t *option) const
                                                      {
                                                        TMatrixDSparse tmp;
                                                        this->GetSub(row_lwb,row_upb,col_lwb,col_upb,tmp,option);
                                                        return tmp;
                                                      }

inline Double_t TMatrixDSparse::operator()(Int_t rown,Int_t coln) const
{
  Assert(IsValid());
  if (fNrowIndex > 0 && fRowIndex[fNrowIndex-1] == 0) {
    Error("operator=()(Int_t,Int_t) const","row/col indices are not set");
    printf("fNrowIndex = %d fRowIndex[fNrowIndex-1] = %d\n",fNrowIndex,fRowIndex[fNrowIndex-1]);
    return 0.0;
  }
  const Int_t arown = rown-fRowLwb;
  const Int_t acoln = coln-fColLwb;
  Assert(arown < fNrows && arown >= 0);
  Assert(acoln < fNcols && acoln >= 0);
  const Int_t sIndex = fRowIndex[arown];
  const Int_t eIndex = fRowIndex[arown+1];
  const Int_t index = TMath::BinarySearch(eIndex-sIndex,fColIndex+sIndex,acoln)+sIndex;
  if (index < sIndex || fColIndex[index] != acoln) return 0.0;
  else                                             return fElements[index];
}

TMatrixDSparse  operator+(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator+(const TMatrixDSparse &source1,const TMatrixD       &source2);
TMatrixDSparse  operator+(const TMatrixD       &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator+(const TMatrixDSparse &source ,      Double_t        val    );
TMatrixDSparse  operator+(      Double_t        val    ,const TMatrixDSparse &source );
TMatrixDSparse  operator-(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator-(const TMatrixDSparse &source1,const TMatrixD       &source2);
TMatrixDSparse  operator-(const TMatrixD       &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator-(const TMatrixDSparse &source ,      Double_t        val    );
TMatrixDSparse  operator-(      Double_t        val    ,const TMatrixDSparse &source );
TMatrixDSparse  operator*(const TMatrixDSparse &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator*(const TMatrixDSparse &source1,const TMatrixD       &source2);
TMatrixDSparse  operator*(const TMatrixD       &source1,const TMatrixDSparse &source2);
TMatrixDSparse  operator*(      Double_t        val    ,const TMatrixDSparse &source );
TMatrixDSparse  operator*(const TMatrixDSparse &source,       Double_t        val    );

TMatrixDSparse &Add        (TMatrixDSparse &target,      Double_t         scalar,const TMatrixDSparse &source);
TMatrixDSparse &ElementMult(TMatrixDSparse &target,const TMatrixDSparse  &source);
TMatrixDSparse &ElementDiv (TMatrixDSparse &target,const TMatrixDSparse  &source);

Bool_t AreCompatible(const TMatrixDSparse &m1,const TMatrixDSparse &m2,Int_t verbose=0);

#endif
