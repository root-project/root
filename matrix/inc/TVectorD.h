// @(#)root/matrix:$Name:  $:$Id: TVectorD.h,v 1.6 2001/05/07 18:41:49 rdm Exp $
// Author: Fons Rademakers   03/11/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVectorD
#define ROOT_TVectorD


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
// to return a matrix, return a TLazyMatrix object instead. The         //
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif


class TMatrixD;
class TElementActionD;
class TElementPosActionD;
class TMatrixDRow;
class TMatrixDColumn;
class TMatrixDDiag;

class TVectorD;
TVectorD &operator+=(TVectorD &target, const TVectorD &source);
TVectorD &operator-=(TVectorD &target, const TVectorD &source);
Double_t operator*(const TVectorD &v1, const TVectorD &v2);
TVectorD &Add(TVectorD &target, Double_t scalar, const TVectorD &source);
TVectorD &ElementMult(TVectorD &target, const TVectorD &source);
TVectorD &ElementDiv(TVectorD &target, const TVectorD &source);

Bool_t operator==(const TVectorD &v1, const TVectorD &v2);
void Compare(const TVectorD &im1, const TVectorD &im2);
Bool_t AreCompatible(const TVectorD &v1, const TVectorD &v2);



class TVectorD : public TObject {

friend class TMatrixDRow;
friend class TMatrixDColumn;
friend class TMatrixDDiag;

protected:
   Int_t     fNmem;             //! number of rows in allocated memory (>=fNrows)
   Int_t     fNrows;            // number of rows
   Int_t     fRowLwb;           // lower bound of the row index
   Double_t *fElements;         //[fNrows] elements themselves

   void Allocate(Int_t nrows, Int_t row_lwb = 0);
   void Invalidate() { fNrows = -1; fElements = 0; }

public:
   TVectorD() { Invalidate(); }
   TVectorD(Int_t n);
   TVectorD(Int_t lwb, Int_t upb);
   TVectorD(const TVectorD &another);
#ifndef __CINT__
   TVectorD(Int_t lwb, Int_t upb, Double_t iv1, ...);
#endif

   virtual ~TVectorD();

   void Draw(Option_t *option="");
   void ResizeTo(Int_t n);
   void ResizeTo(Int_t lwb, Int_t upb);
   void ResizeTo(const TVectorD &v);

   Bool_t IsValid() const;

   Double_t &operator()(Int_t index) const;
   Double_t &operator()(Int_t index);

   Int_t GetLwb() const            { return fRowLwb; }
   Int_t GetUpb() const            { return fNrows + fRowLwb - 1; }
   Int_t GetNrows() const          { return fNrows; }
   Int_t GetNoElements() const     { return fNrows; }

   TVectorD &operator=(const TVectorD &source);
   TVectorD &operator=(Double_t val);
   TVectorD &operator=(const TMatrixDRow &mr);
   TVectorD &operator=(const TMatrixDColumn &mc);
   TVectorD &operator=(const TMatrixDDiag &md);
   TVectorD &operator-=(Double_t val);
   TVectorD &operator+=(Double_t val);
   TVectorD &operator*=(Double_t val);
   TVectorD &operator*=(const TMatrixD &a);

   Bool_t operator==(Double_t val) const;
   Bool_t operator!=(Double_t val) const;
   Bool_t operator<(Double_t val) const;
   Bool_t operator<=(Double_t val) const;
   Bool_t operator>(Double_t val) const;
   Bool_t operator>=(Double_t val) const;

   TVectorD &Zero();
   TVectorD &Abs();
   TVectorD &Sqr();
   TVectorD &Sqrt();

   TVectorD &Apply(TElementActionD &action);
   TVectorD &Apply(TElementPosActionD &action);

   Double_t Norm1() const;
   Double_t Norm2Sqr() const;
   Double_t NormInf() const;

   void Print(Option_t *option="") const;

   friend TVectorD &operator+=(TVectorD &target, const TVectorD &source);
   friend TVectorD &operator-=(TVectorD &target, const TVectorD &source);
   friend Double_t operator*(const TVectorD &v1, const TVectorD &v2);
   friend TVectorD &Add(TVectorD &target, Double_t scalar, const TVectorD &source);
   friend TVectorD &ElementMult(TVectorD &target, const TVectorD &source);
   friend TVectorD &ElementDiv(TVectorD &target, const TVectorD &source);

   friend Bool_t operator==(const TVectorD &v1, const TVectorD &v2);
   friend void Compare(const TVectorD &im1, const TVectorD &im2);
   friend Bool_t AreCompatible(const TVectorD &v1, const TVectorD &v2);

   ClassDef(TVectorD,2)  // Vector class with double precision
};


// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails
void VerifyElementValue(const TVectorD &v, Double_t val);
void VerifyVectorIdentity(const TVectorD &v1, const TVectorD &v2);


//----- inlines ----------------------------------------------------------------

#if !defined(R__HPUX) && !defined(R__MACOSX)

#ifndef __CINT__

inline TVectorD::TVectorD(Int_t n)
{
   Allocate(n);
}

inline TVectorD::TVectorD(Int_t lwb, Int_t upb)
{
   Allocate(upb-lwb+1, lwb);
}

inline Bool_t TVectorD::IsValid() const
{
   if (fNrows == -1)
      return kFALSE;
   return kTRUE;
}

inline Bool_t AreCompatible(const TVectorD &v1, const TVectorD &v2)
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

inline TVectorD &TVectorD::operator=(const TVectorD &source)
{
   if (this != &source && AreCompatible(*this, source)) {
      TObject::operator=(source);
      memcpy(fElements, source.fElements, fNrows*sizeof(Double_t));
   }
   return *this;
}

inline TVectorD::TVectorD(const TVectorD &another)
{
   if (another.IsValid()) {
      Allocate(another.GetUpb()-another.GetLwb()+1, another.GetLwb());
      *this = another;
   } else
      Error("TVectorD(const TVectorD&)", "other vector is not valid");
}

inline void TVectorD::ResizeTo(Int_t n)
{
   TVectorD::ResizeTo(0,n-1);
}

inline void TVectorD::ResizeTo(const TVectorD &v)
{
   TVectorD::ResizeTo(v.GetLwb(), v.GetUpb());
}

inline Double_t &TVectorD::operator()(Int_t ind) const
{
   static Double_t err;
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

inline Double_t &TVectorD::operator()(Int_t index)
{
   return (Double_t&)((*(const TVectorD *)this)(index));
}

inline TVectorD &TVectorD::Zero()
{
   if (!IsValid())
      Error("Zero", "vector not initialized");
   else
      memset(fElements, 0, fNrows*sizeof(Double_t));
   return *this;
}

#endif

#endif

#endif
