// @(#)root/matrix:$Name:  $:$Id: TMatrix.h,v 1.15 2002/07/06 15:55:38 brun Exp $
// Authors: Oleg E. Kiselyov, Fons Rademakers   03/11/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrix
#define ROOT_TMatrix


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Linear Algebra Package                                               //
//                                                                      //
// The present package implements all the basic algorithms dealing      //
// with vectors, matrices, matrix columns, rows, diagonals, etc.        //
//                                                                      //
// Matrix elements are arranged in memory in a COLUMN-wise              //
// fashion (in FORTRAN's spirit). In fact, it makes it very easy to     //
// feed the matrices to FORTRAN procedures, which implement more        //
// elaborate algorithms.                                                //
//                                                                      //
// Unless otherwise specified, matrix and vector indices always start   //
// with 0, spanning up to the specified limit-1.                        //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrix A(TMatrix::kTransposed,B);" and     //
// other fancy constructors as much as possible. If one really needs    //
// to return a matrix, return a TLazyMatrix object instead. The         //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrix m = THaarMatrix(5);" and _is_ efficient.                    //
//                                                                      //
// For usage examples see $ROOTSYS/test/vmatrix.cxx and vvector.cxx     //
// and also:                                                            //
// http://root.cern.ch/root/html/TMatrix.html#TMatrix:description       //
//                                                                      //
// The implementation is based on original code by                      //
// Oleg E. Kiselyov (oleg@pobox.com).                                   //
// Several additions/optimisations by  Eddy Offermann <eddy@rentec.com> //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVector
#include "TVector.h"
#endif

class TMatrix;
class TLazyMatrix;
class TMatrixRow;
class TMatrixColumn;
class TMatrixDiag;
class TMatrixPivoting;

TMatrix &operator+=(TMatrix &target, const TMatrix &source);
TMatrix &operator-=(TMatrix &target, const TMatrix &source);
TMatrix  operator+(const TMatrix &source1, const TMatrix &source2);
TMatrix  operator-(const TMatrix &source1, const TMatrix &source2);
TMatrix  operator*(const TMatrix &source1, const TMatrix &source2);
TMatrix &Add(TMatrix &target, Double_t scalar, const TMatrix &source);
TMatrix &ElementMult(TMatrix &target, const TMatrix &source);
TMatrix &ElementDiv(TMatrix &target, const TMatrix &source);
Bool_t   operator==(const TMatrix &im1, const TMatrix &im2);
void     Compare(const TMatrix &im1, const TMatrix &im2);
Bool_t   AreCompatible(const TMatrix &im1, const TMatrix &im2);
Double_t E2Norm(const TMatrix &m1, const TMatrix &m2);


class TMatrix : public TObject {

friend class TVector;
friend class TMatrixRow;
friend class TMatrixColumn;
friend class TMatrixDiag;
friend class TMatrixPivoting;

protected:
   Int_t     fNrows;            // number of rows
   Int_t     fNcols;            // number of columns
   Int_t     fNelems;           // number of elements in matrix
   Int_t     fRowLwb;           // lower bound of the row index
   Int_t     fColLwb;           // lower bound of the col index
   Real_t   *fElements;	        //[fNelems] elements themselves
   Real_t  **fIndex;            //! index[i] = &matrix(0,i) (col index)

   void Allocate(Int_t nrows, Int_t ncols, Int_t row_lwb = 0, Int_t col_lwb = 0);
   void Invalidate() { fNrows = fNcols = fNelems = -1; fElements = 0; fIndex = 0; }

   Int_t Pdcholesky(const Real_t *a, Real_t *u, const Int_t n);
   void  MakeTridiagonal(TMatrix &a,TVector &d,TVector &e);
   void  MakeEigenVectors(TVector &d,TVector &e,TMatrix &z);
   void  EigenSort(TMatrix &eigenVectors,TVector &eigenValues);

   // Elementary constructors
   void Transpose(const TMatrix &m);
   void Invert(const TMatrix &m);
   void InvertPosDef(const TMatrix &m);
   void AMultB(const TMatrix &a, const TMatrix &b);
   void AtMultB(const TMatrix &a, const TMatrix &b);

   friend void MakeHaarMatrix(TMatrix &m);

public:
   enum EMatrixCreatorsOp1 { kZero, kUnit, kTransposed, kInverted, kInvertedPosDef };
   enum EMatrixCreatorsOp2 { kMult, kTransposeMult, kInvMult, kInvPosDefMult, kAtBA };

   TMatrix() { Invalidate(); }
   TMatrix(Int_t nrows, Int_t ncols);
   TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   TMatrix(Int_t nrows, Int_t ncols, const Float_t *elements, Option_t *option="");
   TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb,
           const Float_t *elements, Option_t *option="");
   TMatrix(const TMatrix &another);
   TMatrix(EMatrixCreatorsOp1 op, const TMatrix &prototype);
   TMatrix(const TMatrix &a, EMatrixCreatorsOp2 op, const TMatrix &b);
   TMatrix(const TLazyMatrix &lazy_constructor);

   virtual ~TMatrix();

   void Draw(Option_t *option="");  // *MENU*
   void ResizeTo(Int_t nrows, Int_t ncols);
   void ResizeTo(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   void ResizeTo(const TMatrix &m);

   Bool_t IsValid() const;
   Bool_t IsSymmetric() const;

   Int_t    GetRowLwb()     const { return fRowLwb; }
   Int_t    GetRowUpb()     const { return fNrows+fRowLwb-1; }
   Int_t    GetNrows()      const { return fNrows; }
   Int_t    GetColLwb()     const { return fColLwb; }
   Int_t    GetColUpb()     const { return fNcols+fColLwb-1; }
   Int_t    GetNcols()      const { return fNcols; }
   Int_t    GetNoElements() const { return fNelems; }
   Float_t *GetElements()         { return fElements; }
   void     GetElements(Float_t *elements, Option_t *option="") const;
   void     SetElements(const Float_t *elements, Option_t *option="");

   const Real_t &operator()(Int_t rown, Int_t coln) const;
   Real_t &operator()(Int_t rown, Int_t coln);
   const TMatrixRow operator[](Int_t rown) const;
   TMatrixRow operator[](Int_t rown);

   TMatrix &operator=(const TMatrix &source);
   TMatrix &operator=(const TLazyMatrix &source);
   TMatrix &operator=(Real_t val);
   TMatrix &operator-=(Double_t val);
   TMatrix &operator+=(Double_t val);
   TMatrix &operator*=(Double_t val);

   Bool_t operator==(Real_t val) const;
   Bool_t operator!=(Real_t val) const;
   Bool_t operator<(Real_t val) const;
   Bool_t operator<=(Real_t val) const;
   Bool_t operator>(Real_t val) const;
   Bool_t operator>=(Real_t val) const;

   TMatrix &Zero();
   TMatrix &Abs();
   TMatrix &Sqr();
   TMatrix &Sqrt();

   TMatrix &Apply(TElementAction &action);
   TMatrix &Apply(TElementPosAction &action);

   TMatrix &Invert(Double_t *determ_ptr = 0);
   TMatrix &InvertPosDef();

   TMatrix EigenVectors(TVector &eigenValues);

   TMatrix &MakeSymmetric();
   TMatrix &UnitMatrix();
   TMatrix &HilbertMatrix();

   TMatrix &operator*=(const TMatrix &source);
   TMatrix &operator*=(const TMatrixDiag &diag);
   TMatrix &operator/=(const TMatrixDiag &diag);
   TMatrix &operator*=(const TMatrixRow &diag);
   TMatrix &operator/=(const TMatrixRow &diag);
   TMatrix &operator*=(const TMatrixColumn &diag);
   TMatrix &operator/=(const TMatrixColumn &diag);

   void Mult(const TMatrix &a, const TMatrix &b);

   Double_t RowNorm() const;
   Double_t NormInf() const { return RowNorm(); }
   Double_t ColNorm() const;
   Double_t Norm1() const { return ColNorm(); }
   Double_t E2Norm() const;
   TMatrix &NormByDiag(const TVector &v, Option_t *option="D");
   TMatrix &NormByColumn(const TVector &v, Option_t *option="D");
   TMatrix &NormByRow(const TVector &v, Option_t *option="D");

   Double_t Determinant() const;

   void Print(Option_t *option="") const;  // *MENU*

   friend TMatrix &operator+=(TMatrix &target, const TMatrix &source);
   friend TMatrix &operator-=(TMatrix &target, const TMatrix &source);
   friend TMatrix &Add(TMatrix &target, Double_t scalar, const TMatrix &source);
   friend TMatrix &ElementMult(TMatrix &target, const TMatrix &source);
   friend TMatrix &ElementDiv(TMatrix &target, const TMatrix &source);

   friend TMatrix  operator+(const TMatrix &source1, const TMatrix &source2);
   friend TMatrix  operator-(const TMatrix &source1, const TMatrix &source2);
   friend TMatrix  operator*(const TMatrix &source1, const TMatrix &source2);

   friend Bool_t operator==(const TMatrix &im1, const TMatrix &im2);
   friend void Compare(const TMatrix &im1, const TMatrix &im2);
   friend Bool_t AreCompatible(const TMatrix &im1, const TMatrix &im2);
   friend Double_t E2Norm(const TMatrix &m1, const TMatrix &m2);

   ClassDef(TMatrix,2)  // Matrix class
};


// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails
void VerifyElementValue(const TMatrix &m, Real_t val);
void VerifyMatrixIdentity(const TMatrix &m1, const TMatrix &m2);


#if !defined(R__HPUX) && !defined(R__MACOSX)
inline Bool_t TMatrix::IsValid() const
   { if (fNrows == -1) return kFALSE; return kTRUE; }
#endif

#ifndef ROOT_TMatrixUtils
#include "TMatrixUtils.h"
#endif


//----- inlines ----------------------------------------------------------------

#if !defined(R__HPUX) && !defined(R__MACOSX)

#ifndef __CINT__

inline TMatrix::TMatrix(Int_t no_rows, Int_t no_cols)
{
   Allocate(no_rows, no_cols);
}

inline TMatrix::TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
{
   Allocate(row_upb-row_lwb+1, col_upb-col_lwb+1, row_lwb, col_lwb);
}

inline void TMatrix::SetElements(const Float_t *elements, Option_t *option)
{
  if (!IsValid()) {
    Error("SetElements", "matrix is not initialized");
    return;
  }

  TString opt = option;
  opt.ToUpper();

  if (opt.Contains("F"))
    memcpy(fElements,elements,fNelems*sizeof(Float_t));
  else
  {
    for (Int_t irow = 0; irow < fNrows; irow++)
    {
      for (Int_t icol = 0; icol < fNcols; icol++)
        fElements[irow+icol*fNrows] = elements[irow*fNcols+icol];
    }
  }
}

inline TMatrix::TMatrix(Int_t no_rows, Int_t no_cols,
                        const Float_t *elements, Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]

  Allocate(no_rows, no_cols);
  SetElements(elements,option);
}

inline TMatrix::TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb,
                        const Float_t *elements, Option_t *option)
{
  Allocate(row_upb-row_lwb+1, col_upb-col_lwb+1, row_lwb, col_lwb);
  SetElements(elements,option);
}

inline void TMatrix::GetElements(Float_t *elements, Option_t *option) const
{
  if (!IsValid()) {
    Error("GetElements", "matrix is not initialized");
    return;
  }

  TString opt = option;
  opt.ToUpper();

  if (opt.Contains("F"))
    memcpy(elements,fElements,fNelems*sizeof(Float_t));
  else
  {
    for (Int_t irow = 0; irow < fNrows; irow++)
    {
      for (Int_t icol = 0; icol < fNcols; icol++)
        elements[irow+icol*fNrows] = fElements[irow*fNcols+icol];
    }
  }
}

inline TMatrix::TMatrix(const TLazyMatrix &lazy_constructor)
{
   Allocate(lazy_constructor.fRowUpb-lazy_constructor.fRowLwb+1,
            lazy_constructor.fColUpb-lazy_constructor.fColLwb+1,
            lazy_constructor.fRowLwb, lazy_constructor.fColLwb);
  lazy_constructor.FillIn(*this);
}

inline TMatrix &TMatrix::operator=(const TLazyMatrix &lazy_constructor)
{
   if (!IsValid()) {
      Error("operator=(const TLazyMatrix&)", "matrix is not initialized");
      return *this;
   }
   if (lazy_constructor.fRowUpb != GetRowUpb() ||
       lazy_constructor.fColUpb != GetColUpb() ||
       lazy_constructor.fRowLwb != GetRowLwb() ||
       lazy_constructor.fColLwb != GetColLwb()) {
      Error("operator=(const TLazyMatrix&)", "matrix is incompatible with "
            "the assigned Lazy matrix");
      return *this;
   }

   lazy_constructor.FillIn(*this);
   return *this;
}

inline Bool_t AreCompatible(const TMatrix &im1, const TMatrix &im2)
{
   if (!im1.IsValid()) {
      ::Error("AreCompatible", "matrix 1 not initialized");
      return kFALSE;
   }
   if (!im2.IsValid()) {
      ::Error("AreCompatible", "matrix 2 not initialized");
      return kFALSE;
   }

   if (im1.fNrows  != im2.fNrows  || im1.fNcols  != im2.fNcols ||
       im1.fRowLwb != im2.fRowLwb || im1.fColLwb != im2.fColLwb) {
      ::Error("AreCompatible", "matrices 1 and 2 not compatible");
      return kFALSE;
   }

   return kTRUE;
}

inline TMatrix &TMatrix::operator=(const TMatrix &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNelems*sizeof(Real_t));
   }
   return *this;
}

inline TMatrix::TMatrix(const TMatrix &another) : TObject()
{
   if (another.IsValid()) {
      Allocate(another.fNrows, another.fNcols, another.fRowLwb, another.fColLwb);
      *this = another;
   } else
      Error("TMatrix(const TMatrix&)", "other matrix is not valid");
}

inline void TMatrix::ResizeTo(const TMatrix &m)
{
   ResizeTo(m.GetRowLwb(), m.GetRowUpb(), m.GetColLwb(), m.GetColUpb());
}

inline const Real_t &TMatrix::operator()(int rown, int coln) const
{
   static Real_t err;
   err = 0.0;

   if (!IsValid()) {
      Error("operator()", "matrix is not initialized");
      return err;
   }

   Int_t arown = rown - fRowLwb;          // Effective indices
   Int_t acoln = coln - fColLwb;

   if (arown >= fNrows || arown < 0) {
      Error("operator()", "row index %d is out of matrix boundaries [%d,%d]",
            rown, fRowLwb, fNrows+fRowLwb-1);
      return err;
   }
   if (acoln >= fNcols || acoln < 0) {
      Error("operator()", "col index %d is out of matrix boundaries [%d,%d]",
            coln, fColLwb, fNcols+fColLwb-1);
      return err;
   }

   return (fIndex[acoln])[arown];
}

inline Real_t &TMatrix::operator()(Int_t rown, Int_t coln)
{
   return (Real_t&)((*(const TMatrix *)this)(rown,coln));
}

inline const TMatrixRow TMatrix::operator[](int rown) const
{
   return TMatrixRow(*this,rown);
}

inline TMatrixRow TMatrix::operator[](int rown)
{
   return TMatrixRow(*this,rown);
}

inline TMatrix &TMatrix::Zero()
{
   if (!IsValid())
      Error("Zero", "matrix not initialized");
   else
      memset(fElements, 0, fNelems*sizeof(Real_t));
   return *this;
}

inline TMatrix &TMatrix::Apply(TElementAction &action)
{
   if (!IsValid())
      Error("Apply(TElementAction&)", "matrix not initialized");
   else
      for (Real_t *ep = fElements; ep < fElements+fNelems; ep++)
         action.Operation(*ep);
   return *this;
}

#endif

#endif

#endif
