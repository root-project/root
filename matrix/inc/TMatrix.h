// @(#)root/matrix:$Name:  $:$Id: TMatrix.h,v 1.9 2001/10/10 06:25:56 brun Exp $
// Author: Fons Rademakers   03/11/97

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
   TMatrix(const TMatrix &another);
   TMatrix(EMatrixCreatorsOp1 op, const TMatrix &prototype);
   TMatrix(const TMatrix &a, EMatrixCreatorsOp2 op, const TMatrix &b);
   TMatrix(const TLazyMatrix &lazy_constructor);

   virtual ~TMatrix();

   void Draw(Option_t *option="");
   void ResizeTo(Int_t nrows, Int_t ncols);
   void ResizeTo(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   void ResizeTo(const TMatrix &m);

   Bool_t IsValid() const;

   Int_t GetRowLwb() const     { return fRowLwb; }
   Int_t GetRowUpb() const     { return fNrows+fRowLwb-1; }
   Int_t GetNrows() const      { return fNrows; }
   Int_t GetColLwb() const     { return fColLwb; }
   Int_t GetColUpb() const     { return fNcols+fColLwb-1; }
   Int_t GetNcols() const      { return fNcols; }
   Int_t GetNoElements() const { return fNelems; }

   const Real_t &operator()(Int_t rown, Int_t coln) const;
   Real_t &operator()(Int_t rown, Int_t coln);

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

   TMatrix &UnitMatrix();
   TMatrix &HilbertMatrix();

   TMatrix &operator*=(const TMatrix &source);
   TMatrix &operator*=(const TMatrixDiag &diag);

   void Mult(const TMatrix &a, const TMatrix &b);

   Double_t RowNorm() const;
   Double_t NormInf() const { return RowNorm(); }
   Double_t ColNorm() const;
   Double_t Norm1() const { return ColNorm(); }
   Double_t E2Norm() const;

   Double_t Determinant() const;

   void Print(Option_t *option="") const;

   friend TMatrix &operator+=(TMatrix &target, const TMatrix &source);
   friend TMatrix &operator-=(TMatrix &target, const TMatrix &source);
   friend TMatrix &Add(TMatrix &target, Double_t scalar, const TMatrix &source);
   friend TMatrix &ElementMult(TMatrix &target, const TMatrix &source);
   friend TMatrix &ElementDiv(TMatrix &target, const TMatrix &source);

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
