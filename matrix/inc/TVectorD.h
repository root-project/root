// @(#)root/matrix:$Name:  $:$Id: TVectorD.h,v 1.30 2004/04/15 09:21:50 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

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
// TVectorD                                                             //
//                                                                      //
// Vectors in the linear algebra package                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMatrixDSym
#include "TMatrixDSym.h"
#endif
#ifndef ROOT_TMatrixDSparse
#include "TMatrixDSparse.h"
#endif

class TVectorD : public TObject {

protected:
  Int_t     fNrows;                // number of rows
  Int_t     fRowLwb;               // lower bound of the row index
  Double_t *fElements;             //[fNrows] elements themselves

  enum {kSizeMax = 5};             // size data container on stack, see New_m(),Delete_m()
  Double_t  fDataStack[kSizeMax];  //! data container
  Bool_t    fIsOwner;              //!default kTRUE, when Use array kFALSE

  Double_t* New_m   (Int_t size);
  void      Delete_m(Int_t size,Double_t*&);
  Int_t     Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  void Allocate  (Int_t nrows,Int_t row_lwb = 0,Int_t init = 0);

public:
  TVectorD() { fIsOwner = kTRUE; fElements = 0; fNrows = 0; fRowLwb = 0; }
  explicit TVectorD(Int_t n);
  TVectorD(Int_t lwb,Int_t upb);
  TVectorD(Int_t n,const Double_t *elements);
  TVectorD(Int_t lwb,Int_t upb,const Double_t *elements);
  TVectorD(const TVectorD             &another);
  TVectorD(const TVectorF             &another);
  TVectorD(const TMatrixDRow_const    &mr);
  TVectorD(const TMatrixDColumn_const &mc);
  TVectorD(const TMatrixDDiag_const   &md);
#ifndef __CINT__
  TVectorD(Int_t lwb,Int_t upb,Double_t iv1, ...);
#endif
  virtual ~TVectorD() { Clear(); }

  inline          Int_t     GetLwb       () const { return fRowLwb; }
  inline          Int_t     GetUpb       () const { return fNrows+fRowLwb-1; }
  inline          Int_t     GetNrows     () const { return fNrows; }
  inline          Int_t     GetNoElements() const { return fNrows; }

  inline          Double_t *GetMatrixArray  ()       { return fElements; }
  inline const    Double_t *GetMatrixArray  () const { return fElements; }

  inline void     Invalidate () { fNrows = -1; }
  inline Bool_t   IsValid    () const { if (fNrows == -1) return kFALSE; return kTRUE; }
  inline Bool_t   IsOwner    () const { return fIsOwner; }
  inline void     SetElements(const Double_t *elements) { Assert(IsValid());
                                                          memcpy(fElements,elements,fNrows*sizeof(Double_t)); }
  inline void     Shift      (Int_t row_shift)   { fRowLwb += row_shift; }
         void     ResizeTo   (Int_t lwb,Int_t upb);
  inline void     ResizeTo   (Int_t n)           { ResizeTo(0,n-1); }
  inline void     ResizeTo   (const TVectorD &v) { ResizeTo(v.GetLwb(),v.GetUpb()); }

         void     Use        (Int_t n,Double_t *data);
         void     Use        (Int_t lwb,Int_t upb,Double_t *data);
         TVectorD GetSub     (Int_t row_lwb,Int_t row_upb,Option_t *option="S") const;
         void     SetSub     (Int_t row_lwb,const TVectorD &source);

  TVectorD &Zero();
  TVectorD &Abs ();
  TVectorD &Sqr ();
  TVectorD &Sqrt();
  TVectorD &Invert();
  TVectorD &SelectNonZeros(const TVectorD &select);

  Double_t Norm1   () const;
  Double_t Norm2Sqr() const;
  Double_t NormInf () const;
  Int_t    NonZeros() const;
  Double_t Sum     () const;
  Double_t Min     () const;
  Double_t Max     () const;

  inline const Double_t &operator()(Int_t index) const;
  inline       Double_t &operator()(Int_t index)       { return (Double_t&)((*(const TVectorD *)this)(index)); }
  inline const Double_t &operator[](Int_t index) const { return (Double_t&)((*(const TVectorD *)this)(index)); }
  inline       Double_t &operator[](Int_t index)       { return (Double_t&)((*(const TVectorD *)this)(index)); }

  TVectorD &operator= (const TVectorD                &source);
  TVectorD &operator= (const TVectorF                &source);
  TVectorD &operator= (const TMatrixDRow_const       &mr);
  TVectorD &operator= (const TMatrixDColumn_const    &mc);
  TVectorD &operator= (const TMatrixDDiag_const      &md);
  TVectorD &operator= (const TMatrixDSparseRow_const &md);
  TVectorD &operator= (Double_t val);
  TVectorD &operator+=(Double_t val);
  TVectorD &operator-=(Double_t val);
  TVectorD &operator*=(Double_t val);

  TVectorD &operator+=(const TVectorD       &source);
  TVectorD &operator-=(const TVectorD       &source);
  TVectorD &operator*=(const TMatrixD       &a);
  TVectorD &operator*=(const TMatrixDSym    &a);
  TVectorD &operator*=(const TMatrixDSparse &a);

  Bool_t operator==(Double_t val) const;
  Bool_t operator!=(Double_t val) const;
  Bool_t operator< (Double_t val) const;
  Bool_t operator<=(Double_t val) const;
  Bool_t operator> (Double_t val) const;
  Bool_t operator>=(Double_t val) const;
  Bool_t MatchesNonZeroPattern(const TVectorD &select);
  Bool_t SomePositive         (const TVectorD &select);
  void   AddSomeConstant      (Double_t val,const TVectorD &select);

  void   Randomize(Double_t alpha,Double_t beta,Double_t &seed);

  TVectorD &Apply(const TElementActionD    &action);
  TVectorD &Apply(const TElementPosActionD &action);

  void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNrows,fElements); fNrows = 0; }
  void Draw (Option_t *option=""); // *MENU*
  void Print(Option_t *option="") const;  // *MENU*

  ClassDef(TVectorD,2)  // Vector class with double precision
};

inline const Double_t &TVectorD::operator()(Int_t ind) const
{
  // Access a vector element.

  Assert(IsValid());
  const Int_t aind = ind-fRowLwb;
  Assert(aind < fNrows && aind >= 0);

  return fElements[aind];
}

Bool_t    operator==    (const TVectorD       &source1,const TVectorD &source2);
TVectorD  operator+     (const TVectorD       &source1,const TVectorD &source2);
TVectorD  operator-     (const TVectorD       &source1,const TVectorD &source2);
Double_t  operator*     (const TVectorD       &source1,const TVectorD &source2);
TVectorD  operator*     (const TMatrixD       &a,      const TVectorD &source);
TVectorD  operator*     (const TMatrixDSym    &a,      const TVectorD &source);
TVectorD  operator*     (const TMatrixDSparse &a,      const TVectorD &source);
TVectorD  operator*     (      Double_t        val,    const TVectorD &source);
TVectorD  &Add          (      TVectorD       &target,       Double_t  scalar,const TVectorD &source);
TVectorD  &AddElemMult  (      TVectorD       &target,       Double_t  scalar,const TVectorD &source1,
                         const TVectorD       &source2);
TVectorD  &AddElemMult  (      TVectorD       &target,       Double_t  scalar,const TVectorD &source1,
                         const TVectorD       &source2,const TVectorD &select);
TVectorD  &AddElemDiv   (      TVectorD       &target,       Double_t  scalar,const TVectorD &source1,
                         const TVectorD       &source2);
TVectorD  &AddElemDiv   (      TVectorD       &target,       Double_t  scalar,const TVectorD &source1,
                         const TVectorD       &source2,const TVectorD &select);
TVectorD  &ElementMult  (      TVectorD       &target, const TVectorD &source);
TVectorD  &ElementMult  (      TVectorD       &target, const TVectorD &source,const TVectorD &select);
TVectorD  &ElementDiv   (      TVectorD       &target, const TVectorD &source);
TVectorD  &ElementDiv   (      TVectorD       &target, const TVectorD &source,const TVectorD &select);
Bool_t     AreCompatible(const TVectorD       &source1,const TVectorD &source2,Int_t verbose=0);
Bool_t     AreCompatible(const TVectorD       &source1,const TVectorF &source2,Int_t verbose=0);
void       Compare      (const TVectorD       &source1,const TVectorD &source2);

// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails

Bool_t VerifyVectorValue   (const TVectorD &m,Double_t val,
                            Int_t verbose=1,Double_t maxDevAllow=DBL_EPSILON);
Bool_t VerifyVectorIdentity(const TVectorD &m1,const TVectorD &m2,
                            Int_t verbose=1,Double_t maxDevAllow=DBL_EPSILON);
#endif
