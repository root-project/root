// @(#)root/matrix:$Name:  $:$Id: TVectorF.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVectorF
#define ROOT_TVectorF

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVectorF                                                             //
//                                                                      //
// Vectors in the linear algebra package                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixFBase
#include "TMatrixFBase.h"
#endif

class TVectorF : public TObject {

protected:
  Int_t     fNrows;                // number of rows
  Int_t     fRowLwb;               // lower bound of the row index
  Float_t  *fElements;             //[fNrows] elements themselves

  enum {kSizeMax = 5};             // size data container on stack, see New_m(),Delete_m()
  Float_t   fDataStack[kSizeMax];  //! data container
  Bool_t    fIsOwner;              //!default kTRUE, when Adopt array kFALSE

  Float_t*  New_m   (Int_t size);
  void      Delete_m(Int_t size,Float_t*);
  Int_t     Memcpy_m(Float_t *newp,const Float_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  void Allocate  (Int_t nrows,Int_t row_lwb = 0,Int_t init = 0);

public:
  TVectorF() { fIsOwner = kTRUE; fElements = 0; Invalidate(); }
  explicit TVectorF(Int_t n);
  TVectorF(Int_t lwb,Int_t upb);
  TVectorF(Int_t n,const Float_t *elements);
  TVectorF(Int_t lwb,Int_t upb,const Float_t *elements);
  TVectorF(const TVectorF             &another);
  TVectorF(const TMatrixFRow_const    &mr);
  TVectorF(const TMatrixFColumn_const &mc);
  TVectorF(const TMatrixFDiag_const   &md);
#ifndef __CINT__
  TVectorF(Int_t lwb,Int_t upb,Float_t iv1, ...);
#endif
  virtual ~TVectorF() { Clear(); Invalidate(); }

  inline          Int_t     GetLwb       () const { return fRowLwb; }
  inline          Int_t     GetUpb       () const { return fNrows+fRowLwb-1; }
  inline          Int_t     GetNrows     () const { return fNrows; }
  inline          Int_t     GetNoElements() const { return fNrows; }

  inline          Float_t  *GetElements  ()       { return fElements; }
  inline const    Float_t  *GetElements  () const { return fElements; }

  inline void     Invalidate () { fNrows = -1; }
  inline Bool_t   IsValid    () const { if (fNrows == -1) return kFALSE; return kTRUE; }
  inline void     SetElements(const Float_t *elements) { Assert(IsValid());
                                                          memcpy(fElements,elements,fNrows*sizeof(Float_t)); }
         void     ResizeTo   (Int_t lwb,Int_t upb);
  inline void     ResizeTo   (Int_t n)       { ResizeTo(0,n-1); }
  inline void     ResizeTo   (const TVectorF &v) { ResizeTo(v.GetLwb(),v.GetUpb()); }

         void     Adopt      (Int_t n,Float_t *data);
         void     Adopt      (Int_t lwb,Int_t upb,Float_t *data);
         TVectorF GetSub     (Int_t row_lwb,Int_t row_upb,Option_t *option="S") const;
         void     SetSub     (Int_t row_lwb,const TVectorF &source);

  TVectorF &Zero();
  TVectorF &Abs ();
  TVectorF &Sqr ();
  TVectorF &Sqrt();

  Float_t Norm1   () const;
  Float_t Norm2Sqr() const;
  Float_t NormInf () const;

  inline const Float_t &operator()(Int_t index) const;
  inline       Float_t &operator()(Int_t index)       { return (Float_t&)((*(const TVectorF *)this)(index)); }
  inline const Float_t &operator[](Int_t index) const { return (Float_t&)((*(const TVectorF *)this)(index)); }
  inline       Float_t &operator[](Int_t index)       { return (Float_t&)((*(const TVectorF *)this)(index)); }

  TVectorF &operator= (const TVectorF             &source);
  TVectorF &operator= (const TMatrixFRow_const    &mr);
  TVectorF &operator= (const TMatrixFColumn_const &mc);
  TVectorF &operator= (const TMatrixFDiag_const   &md);
  TVectorF &operator= (Float_t val);
  TVectorF &operator+=(Float_t val);
  TVectorF &operator-=(Float_t val);
  TVectorF &operator*=(Float_t val);

  TVectorF &operator+=(const TVectorF    &source);
  TVectorF &operator-=(const TVectorF    &source);
  TVectorF &operator*=(const TMatrixF    &a);
  TVectorF &operator*=(const TMatrixFSym &a);

  Bool_t operator==(Float_t val) const;
  Bool_t operator!=(Float_t val) const;
  Bool_t operator< (Float_t val) const;
  Bool_t operator<=(Float_t val) const;
  Bool_t operator> (Float_t val) const;
  Bool_t operator>=(Float_t val) const;

  TVectorF &Apply(const TElementActionF    &action);
  TVectorF &Apply(const TElementPosActionF &action);

  void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNrows,fElements); }
  void Draw (Option_t *option=""); // *MENU*
  void Print(Option_t *option="") const;  // *MENU*

  friend Bool_t    operator== (const TVectorF    &v1,     const TVectorF &v2);
  friend Float_t   operator*  (const TVectorF    &v1,     const TVectorF &v2);
  friend TVectorF  operator+  (const TVectorF    &source1,const TVectorF &source2);
  friend TVectorF  operator-  (const TVectorF    &source1,const TVectorF &source2);
  friend TVectorF  operator*  (const TMatrixF    &a,      const TVectorF &source);
  friend TVectorF  operator*  (const TMatrixFSym &a,      const TVectorF &source);

  friend TVectorF &Add        (TVectorF &target,Float_t scalar,const TVectorF &source);
  friend TVectorF &ElementMult(TVectorF &target,const TVectorF &source);
  friend TVectorF &ElementDiv (TVectorF &target,const TVectorF &source);

  friend Bool_t AreCompatible(const TVectorF &v1,const TVectorF &v2,Int_t verbose);
  friend void   Compare      (const TVectorF &v1,const TVectorF &v2);

  ClassDef(TVectorF,2)  // Vector class with single precision
};

class TVector : public TVectorF {
public :
  TVector() {}
  explicit TVector(Int_t n)                            : TVectorF(n) {}
  TVector(Int_t lwb,Int_t upb)                         : TVectorF(lwb,upb) {}
  TVector(Int_t n,const Float_t *elements)             : TVectorF(n,elements) {}
  TVector(Int_t lwb,Int_t upb,const Float_t *elements) : TVectorF(lwb,upb,elements) {}
  TVector(const TVectorF             &another)         : TVectorF(another) {}
  TVector(const TMatrixFRow_const    &mr)              : TVectorF(mr) {}
  TVector(const TMatrixFColumn_const &mc)              : TVectorF(mc) {}
  TVector(const TMatrixFDiag_const   &md)              : TVectorF(md) {}

  virtual ~TVector() {}
  ClassDef(TVector,2)  // Vector class with single precision
};

inline const Float_t &TVectorF::operator()(Int_t ind) const
{
  // Access a vector element.

  Assert(IsValid());
  const Int_t aind = ind-fRowLwb;
  Assert(aind < fNrows && aind >= 0);

  return fElements[aind];
}

Bool_t    operator==    (const TVectorF    &source1,const TVectorF &source2);
TVectorF  operator+     (const TVectorF    &source1,const TVectorF &source2);
TVectorF  operator-     (const TVectorF    &source1,const TVectorF &source2);
TVectorF  operator*     (const TMatrixF    &a,      const TVectorF &source);
TVectorF  operator*     (const TMatrixFSym &a,      const TVectorF &source);
Float_t   operator*     (const TVectorF    &source1,const TVectorF &source2);
TVectorF  &Add          (      TVectorF    &target,       Float_t  scalar,const TVectorF &source);
TVectorF  &ElementMult  (      TVectorF    &target, const TVectorF &source);
TVectorF  &ElementDiv   (      TVectorF    &target, const TVectorF &source);
Bool_t     AreCompatible(const TVectorF    &source1,const TVectorF &source2,Int_t verbose=0);
void       Compare      (const TVectorF    &source1,const TVectorF &source2);

// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails

Bool_t VerifyVectorValue   (const TVectorF &m,Float_t val,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);
Bool_t VerifyVectorIdentity(const TVectorF &m1,const TVectorF &m2,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);
#endif
