// @(#)root/matrix:$Name:  $:$Id: TMatrixD.h,v 1.1 2000/06/16 15:15:47 rdm Exp $
// Author: Fons Rademakers   03/11/97

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
// returning matrices. Use "TMatrixD A(TMatrixD::kTransposed,B);" and   //
// other fancy constructors as much as possible. If one really needs    //
// to return a matrix, return a TLazyMatrixD object instead. The        //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrixD m = THaarMatrixD(5);" and _is_ efficient.                  //
//                                                                      //
// For usage examples see $ROOTSYS/test/vmatrix.cxx and vvector.cxx     //
// and also:                                                            //
// http://root.cern.ch/root/html/TMatrixD.html#TMatrixD:description     //
//                                                                      //
// The implementation is based on original code by                      //
// Oleg E. Kiselyov (oleg@pobox.com).                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif

class TLazyMatrixD;



class TMatrixD : public TObject {

friend class TVectorD;
friend class TMatrixDRow;
friend class TMatrixDColumn;
friend class TMatrixDDiag;
friend class TMatrixDPivoting;

protected:
   Int_t      fNrows;            // number of rows
   Int_t      fNcols;            // number of columns
   Int_t      fNelems;           // number of elements in matrix
   Int_t      fRowLwb;           // lower bound of the row index
   Int_t      fColLwb;           // lower bound of the col index
   Double_t  *fElements;         // elements themselves
   Double_t **fIndex;            // index[i] = &matrix(0,i) (col index)

   void Allocate(Int_t nrows, Int_t ncols, Int_t row_lwb = 0, Int_t col_lwb = 0);
   void Invalidate() { fNrows = fNcols = fNelems = -1; fElements = 0; fIndex = 0; }

   // Elementary constructors
   void Transpose(const TMatrixD &m);
   void Invert(const TMatrixD &m);
   void AMultB(const TMatrixD &a, const TMatrixD &b);
   void AtMultB(const TMatrixD &a, const TMatrixD &b);

   friend void MakeHaarMatrixD(TMatrixD &m);

public:
   enum EMatrixCreatorsOp1 { kZero, kUnit, kTransposed, kInverted };
   enum EMatrixCreatorsOp2 { kMult, kTransposeMult, kInvMult, kAtBA };

   TMatrixD() { Invalidate(); }
   TMatrixD(Int_t nrows, Int_t ncols);
   TMatrixD(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   TMatrixD(const TMatrixD &another);
   TMatrixD(EMatrixCreatorsOp1 op, const TMatrixD &prototype);
   TMatrixD(const TMatrixD &a, EMatrixCreatorsOp2 op, const TMatrixD &b);
   TMatrixD(const TLazyMatrixD &lazy_constructor);

   virtual ~TMatrixD();

   void Draw(Option_t *option="");
   void ResizeTo(Int_t nrows, Int_t ncols);
   void ResizeTo(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   void ResizeTo(const TMatrixD &m);

   Bool_t IsValid() const;

   Int_t GetRowLwb() const     { return fRowLwb; }
   Int_t GetRowUpb() const     { return fNrows+fRowLwb-1; }
   Int_t GetNrows() const      { return fNrows; }
   Int_t GetColLwb() const     { return fColLwb; }
   Int_t GetColUpb() const     { return fNcols+fColLwb-1; }
   Int_t GetNcols() const      { return fNcols; }
   Int_t GetNoElements() const { return fNelems; }

   const Double_t &operator()(Int_t rown, Int_t coln) const;
   Double_t &operator()(Int_t rown, Int_t coln);

   TMatrixD &operator=(const TMatrixD &source);
   TMatrixD &operator=(const TLazyMatrixD &source);
   TMatrixD &operator=(Double_t val);
   TMatrixD &operator-=(Double_t val);
   TMatrixD &operator+=(Double_t val);
   TMatrixD &operator*=(Double_t val);

   Bool_t operator==(Double_t val) const;
   Bool_t operator!=(Double_t val) const;
   Bool_t operator<(Double_t val) const;
   Bool_t operator<=(Double_t val) const;
   Bool_t operator>(Double_t val) const;
   Bool_t operator>=(Double_t val) const;

   TMatrixD &Zero();
   TMatrixD &Abs();
   TMatrixD &Sqr();
   TMatrixD &Sqrt();

   TMatrixD &Apply(TElementActionD &action);
   TMatrixD &Apply(TElementPosActionD &action);

   TMatrixD &Invert(Double_t *determ_ptr = 0);

   TMatrixD &UnitMatrix();
   TMatrixD &HilbertMatrix();

   TMatrixD &operator*=(const TMatrixD &source);
   TMatrixD &operator*=(const TMatrixDDiag &diag);

   void Mult(const TMatrixD &a, const TMatrixD &b);

   Double_t RowNorm() const;
   Double_t NormInf() const { return RowNorm(); }
   Double_t ColNorm() const;
   Double_t Norm1() const { return ColNorm(); }
   Double_t E2Norm() const;

   Double_t Determinant() const;

   void Print(Option_t *option="");

   friend TMatrixD &operator+=(TMatrixD &target, const TMatrixD &source);
   friend TMatrixD &operator-=(TMatrixD &target, const TMatrixD &source);
   friend TMatrixD &Add(TMatrixD &target, Double_t scalar, const TMatrixD &source);
   friend TMatrixD &ElementMult(TMatrixD &target, const TMatrixD &source);
   friend TMatrixD &ElementDiv(TMatrixD &target, const TMatrixD &source);

   friend Bool_t operator==(const TMatrixD &im1, const TMatrixD &im2);
   friend void Compare(const TMatrixD &im1, const TMatrixD &im2);
   friend Bool_t AreCompatible(const TMatrixD &im1, const TMatrixD &im2);
   friend Double_t E2Norm(const TMatrixD &m1, const TMatrixD &m2);

   ClassDef(TMatrixD,1)  // Matrix class (double precision)
};


// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails
void VerifyElementValue(const TMatrixD &m, Double_t val);
void VerifyMatrixIdentity(const TMatrixD &m1, const TMatrixD &m2);


#ifndef R__HPUX
inline Bool_t TMatrixD::IsValid() const
   { if (fNrows == -1) return kFALSE; return kTRUE; }
#endif

#ifndef ROOT_TMatrixDUtils
#include "TMatrixDUtils.h"
#endif


//----- inlines ----------------------------------------------------------------

#ifndef R__HPUX

#ifndef __CINT__

inline TMatrixD::TMatrixD(Int_t no_rows, Int_t no_cols)
{
   Allocate(no_rows, no_cols);
}

inline TMatrixD::TMatrixD(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
{
   Allocate(row_upb-row_lwb+1, col_upb-col_lwb+1, row_lwb, col_lwb);
}

inline TMatrixD::TMatrixD(const TLazyMatrixD &lazy_constructor)
{
   Allocate(lazy_constructor.fRowUpb-lazy_constructor.fRowLwb+1,
            lazy_constructor.fColUpb-lazy_constructor.fColLwb+1,
            lazy_constructor.fRowLwb, lazy_constructor.fColLwb);
  lazy_constructor.FillIn(*this);
}

inline TMatrixD &TMatrixD::operator=(const TLazyMatrixD &lazy_constructor)
{
   if (!IsValid()) {
      Error("operator=(const TLazyMatrixD&)", "matrix is not initialized");
      return *this;
   }
   if (lazy_constructor.fRowUpb != GetRowUpb() ||
       lazy_constructor.fColUpb != GetColUpb() ||
       lazy_constructor.fRowLwb != GetRowLwb() ||
       lazy_constructor.fColLwb != GetColLwb()) {
      Error("operator=(const TLazyMatrixD&)", "matrix is incompatible with "
            "the assigned Lazy matrix");
      return *this;
   }

   lazy_constructor.FillIn(*this);
   return *this;
}

inline Bool_t AreCompatible(const TMatrixD &im1, const TMatrixD &im2)
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

inline TMatrixD &TMatrixD::operator=(const TMatrixD &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNelems*sizeof(Double_t));
   }
   return *this;
}

inline TMatrixD::TMatrixD(const TMatrixD &another)
{
   if (another.IsValid()) {
      Allocate(another.fNrows, another.fNcols, another.fRowLwb, another.fColLwb);
      *this = another;
   } else
      Error("TMatrixD(const TMatrixD&)", "other matrix is not valid");
}

inline void TMatrixD::ResizeTo(const TMatrixD &m)
{
   ResizeTo(m.GetRowLwb(), m.GetRowUpb(), m.GetColLwb(), m.GetColUpb());
}

inline const Double_t &TMatrixD::operator()(int rown, int coln) const
{
   static Double_t err;
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

inline Double_t &TMatrixD::operator()(Int_t rown, Int_t coln)
{
   return (Double_t&)((*(const TMatrixD *)this)(rown,coln));
}

inline TMatrixD &TMatrixD::Zero()
{
   if (!IsValid())
      Error("Zero", "matrix not initialized");
   else
      memset(fElements, 0, fNelems*sizeof(Double_t));
   return *this;
}

inline TMatrixD &TMatrixD::Apply(TElementActionD &action)
{
   if (!IsValid())
      Error("Apply(TElementActionD&)", "matrix not initialized");
   else
      for (Double_t *ep = fElements; ep < fElements+fNelems; ep++)
         action.Operation(*ep);
   return *this;
}

#endif

#endif

#endif
