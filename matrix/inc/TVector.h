// @(#)root/matrix:$Name:  $:$Id: TVector.h,v 1.7 2001/06/29 17:28:07 brun Exp $
// Author: Fons Rademakers   05/11/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVector
#define ROOT_TVector


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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif


class TVector;
class TMatrix;
class TElementAction;
class TElementPosAction;
class TMatrixRow;
class TMatrixColumn;
class TMatrixDiag;

TVector &operator+=(TVector &target, const TVector &source);
TVector &operator-=(TVector &target, const TVector &source);
Double_t operator*(const TVector &v1, const TVector &v2);
TVector &Add(TVector &target, Double_t scalar, const TVector &source);
TVector &ElementMult(TVector &target, const TVector &source);
TVector &ElementDiv(TVector &target, const TVector &source);
Bool_t   operator==(const TVector &v1, const TVector &v2);
void     Compare(const TVector &im1, const TVector &im2);
Bool_t   AreCompatible(const TVector &v1, const TVector &v2);


class TVector : public TObject {

friend class TMatrixRow;
friend class TMatrixColumn;
friend class TMatrixDiag;

protected:
   Int_t     fNmem;             //! number of rows in allocated memory (>=fNrows)
   Int_t     fNrows;            // number of rows
   Int_t     fRowLwb;           // lower bound of the row index
   Real_t   *fElements;	        //[fNrows] elements themselves

   void Allocate(Int_t nrows, Int_t row_lwb = 0);
   void Invalidate() { fNrows = -1; fElements = 0; }

public:
   TVector() { Invalidate(); }
   TVector(Int_t n);
   TVector(Int_t lwb, Int_t upb);
   TVector(const TVector &another);
#ifndef __CINT__
   TVector(Int_t lwb, Int_t upb, Double_t iv1, ...);
#endif

   virtual ~TVector();

   void Draw(Option_t *option="");
   void ResizeTo(Int_t n);
   void ResizeTo(Int_t lwb, Int_t upb);
   void ResizeTo(const TVector &v);

   Bool_t IsValid() const;

   Real_t &operator()(Int_t index) const;
   Real_t &operator()(Int_t index);

   Int_t GetLwb() const            { return fRowLwb; }
   Int_t GetUpb() const            { return fNrows + fRowLwb - 1; }
   Int_t GetNrows() const          { return fNrows; }
   Int_t GetNoElements() const     { return fNrows; }

   TVector &operator=(const TVector &source);
   TVector &operator=(Real_t val);
   TVector &operator=(const TMatrixRow &mr);
   TVector &operator=(const TMatrixColumn &mc);
   TVector &operator=(const TMatrixDiag &md);
   TVector &operator-=(Double_t val);
   TVector &operator+=(Double_t val);
   TVector &operator*=(Double_t val);
   TVector &operator*=(const TMatrix &a);

   Bool_t operator==(Real_t val) const;
   Bool_t operator!=(Real_t val) const;
   Bool_t operator<(Real_t val) const;
   Bool_t operator<=(Real_t val) const;
   Bool_t operator>(Real_t val) const;
   Bool_t operator>=(Real_t val) const;

   TVector &Zero();
   TVector &Abs();
   TVector &Sqr();
   TVector &Sqrt();

   TVector &Apply(TElementAction &action);
   TVector &Apply(TElementPosAction &action);

   Double_t Norm1() const;
   Double_t Norm2Sqr() const;
   Double_t NormInf() const;

   void Print(Option_t *option="") const;

   friend TVector &operator+=(TVector &target, const TVector &source);
   friend TVector &operator-=(TVector &target, const TVector &source);
   friend Double_t operator*(const TVector &v1, const TVector &v2);
   friend TVector &Add(TVector &target, Double_t scalar, const TVector &source);
   friend TVector &ElementMult(TVector &target, const TVector &source);
   friend TVector &ElementDiv(TVector &target, const TVector &source);

   friend Bool_t operator==(const TVector &v1, const TVector &v2);
   friend void Compare(const TVector &im1, const TVector &im2);
   friend Bool_t AreCompatible(const TVector &v1, const TVector &v2);

   ClassDef(TVector,2)  // Vector class
};


// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails
void VerifyElementValue(const TVector &v, Real_t val);
void VerifyVectorIdentity(const TVector &v1, const TVector &v2);


//----- inlines ----------------------------------------------------------------

#if !defined(R__HPUX) && !defined(R__MACOSX)

#ifndef __CINT__

inline TVector::TVector(Int_t n)
{
   Allocate(n);
}

inline TVector::TVector(Int_t lwb, Int_t upb)
{
   Allocate(upb-lwb+1, lwb);
}

inline Bool_t TVector::IsValid() const
{
   if (fNrows == -1)
      return kFALSE;
   return kTRUE;
}

inline Bool_t AreCompatible(const TVector &v1, const TVector &v2)
{
   if (!v1.IsValid()) {
      ::Error("AreCompatible", "vector 1 not initialized");
      return kFALSE;
   }
   if (!v2.IsValid()) {
      ::Error("AreCompatible", "vector 2 not initialized");
      return kFALSE;
   }

   if (v1.fNrows != v2.fNrows || v1.fRowLwb != v2.fRowLwb) {
      ::Error("AreCompatible", "vectors 1 and 2 not compatible");
      return kFALSE;
   }

   return kTRUE;
}

inline TVector &TVector::operator=(const TVector &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNrows*sizeof(Real_t));
   }
   return *this;
}

inline TVector::TVector(const TVector &another)
{
   if (another.IsValid()) {
      Allocate(another.GetUpb()-another.GetLwb()+1, another.GetLwb());
      *this = another;
   } else
      Error("TVector(const TVector&)", "other vector is not valid");
}

inline void TVector::ResizeTo(Int_t n)
{
   TVector::ResizeTo(0,n-1);
}

inline void TVector::ResizeTo(const TVector &v)
{
   TVector::ResizeTo(v.GetLwb(), v.GetUpb());
}

inline Real_t &TVector::operator()(Int_t ind) const
{
   static Real_t err;
   err = 0.0;

   if (!IsValid()) {
      Error("operator()", "vector is not initialized");
      return err;
   }

   Int_t aind = ind - fRowLwb;
   if (aind >= fNrows || aind < 0) {
      Error("operator()", "requested element %d is out of vector boundaries [%d,%d]",
            ind, fRowLwb, fNrows+fRowLwb-1);
      return err;
   }

   return fElements[aind];
}

inline Real_t &TVector::operator()(Int_t index)
{
   return (Real_t&)((*(const TVector *)this)(index));
}

inline TVector &TVector::Zero()
{
   if (!IsValid())
      Error("Zero", "vector not initialized");
   else
      memset(fElements, 0, fNrows*sizeof(Real_t));
   return *this;
}

#endif

#endif

#endif
