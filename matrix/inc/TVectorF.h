// @(#)root/matrix:$Name:  $:$Id: TVectorF.h,v 1.11 2004/05/27 13:17:41 brun Exp $
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

#ifndef ROOT_TMatrixF
#include "TMatrixF.h"
#endif
#ifndef ROOT_TMatrixFSym
#include "TMatrixFSym.h"
#endif

class TVectorF : public TObject {

protected:
  Int_t     fNrows;                // number of rows
  Int_t     fRowLwb;               // lower bound of the row index
  Float_t  *fElements;             //[fNrows] elements themselves

  enum {kSizeMax = 5};             // size data container on stack, see New_m(),Delete_m()
  Float_t   fDataStack[kSizeMax];  //! data container
  Bool_t    fIsOwner;              //!default kTRUE, when Use array kFALSE

  Float_t*  New_m   (Int_t size);
  void      Delete_m(Int_t size,Float_t*&);
  Int_t     Memcpy_m(Float_t *newp,const Float_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  void Allocate  (Int_t nrows,Int_t row_lwb = 0,Int_t init = 0);

public:

  TVectorF() { SetBit(TMatrixFBase::kStatus); fIsOwner = kTRUE; fElements = 0; fNrows = 0; fRowLwb = 0; }
  explicit TVectorF(Int_t n);
  TVectorF(Int_t lwb,Int_t upb);
  TVectorF(Int_t n,const Float_t *elements);
  TVectorF(Int_t lwb,Int_t upb,const Float_t *elements);
  TVectorF(const TVectorF             &another);
  TVectorF(const TVectorD             &another);
  TVectorF(const TMatrixFRow_const    &mr);
  TVectorF(const TMatrixFColumn_const &mc);
  TVectorF(const TMatrixFDiag_const   &md);
#ifndef __CINT__
  TVectorF(Int_t lwb,Int_t upb,Float_t iv1, ...);
#endif
  virtual ~TVectorF() { Clear(); }

  inline          Int_t     GetLwb       () const { return fRowLwb; }
  inline          Int_t     GetUpb       () const { return fNrows+fRowLwb-1; }
  inline          Int_t     GetNrows     () const { return fNrows; }
  inline          Int_t     GetNoElements() const { return fNrows; }

  inline          Float_t  *GetMatrixArray  ()       { return fElements; }
  inline const    Float_t  *GetMatrixArray  () const { return fElements; }

  inline void     Invalidate ()       { ResetBit(TMatrixFBase::kStatus); }
  inline void     MakeValid  ()       { SetBit(TMatrixFBase::kStatus); }
  inline Bool_t   IsValid    () const { return TestBit(TMatrixFBase::kStatus); }
  inline Bool_t   IsOwner    () const { return fIsOwner; }
  inline void     SetElements(const Float_t *elements) { Assert(IsValid());
                                                          memcpy(fElements,elements,fNrows*sizeof(Float_t)); }
  inline TVectorF &Shift     (Int_t row_shift)   { fRowLwb += row_shift; return *this; }
         TVectorF &ResizeTo  (Int_t lwb,Int_t upb);
  inline TVectorF &ResizeTo  (Int_t n)           { return ResizeTo(0,n-1); }
  inline TVectorF &ResizeTo  (const TVectorF &v) { return ResizeTo(v.GetLwb(),v.GetUpb()); }

         TVectorF &Use       (Int_t n,Float_t *data);
         TVectorF &Use       (Int_t lwb,Int_t upb,Float_t *data);
         TVectorF &Use       (TVectorF &v);
         TVectorF &GetSub    (Int_t row_lwb,Int_t row_upb,TVectorF &target,Option_t *option="S") const;
         TVectorF  GetSub    (Int_t row_lwb,Int_t row_upb,Option_t *option="S") const;
         TVectorF &SetSub    (Int_t row_lwb,const TVectorF &source);

  TVectorF &Zero();
  TVectorF &Abs ();
  TVectorF &Sqr ();
  TVectorF &Sqrt();
  TVectorF &Invert();
  TVectorF &SelectNonZeros(const TVectorF &select);

  Float_t Norm1   () const;
  Float_t Norm2Sqr() const;
  Float_t NormInf () const;
  Int_t   NonZeros() const;
  Float_t Sum     () const;
  Float_t Min     () const;
  Float_t Max     () const;

  inline const Float_t &operator()(Int_t index) const;
  inline       Float_t &operator()(Int_t index)       { return (Float_t&)((*(const TVectorF *)this)(index)); }
  inline const Float_t &operator[](Int_t index) const { return (Float_t&)((*(const TVectorF *)this)(index)); }
  inline       Float_t &operator[](Int_t index)       { return (Float_t&)((*(const TVectorF *)this)(index)); }

  TVectorF &operator= (const TVectorF             &source);
  TVectorF &operator= (const TVectorD             &source);
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
  Bool_t MatchesNonZeroPattern(const TVectorF &select);
  Bool_t SomePositive         (const TVectorF &select);
  void   AddSomeConstant      (Float_t val,const TVectorF &select);

  void   Randomize(Float_t alpha,Float_t beta,Double_t &seed);

  TVectorF &Apply(const TElementActionF    &action);
  TVectorF &Apply(const TElementPosActionF &action);

  void Clear(Option_t * /*option*/ ="") { if (fIsOwner) Delete_m(fNrows,fElements); fNrows = 0; }
  void Draw (Option_t *option=""); // *MENU*
  void Print(Option_t *option="") const;  // *MENU*

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
  TVector(const TVectorD             &another)         : TVectorF(another) {}
  TVector(const TMatrixFRow_const    &mr)              : TVectorF(mr) {}
  TVector(const TMatrixFColumn_const &mc)              : TVectorF(mc) {}
  TVector(const TMatrixFDiag_const   &md)              : TVectorF(md) {}

  virtual ~TVector() {}
  ClassDef(TVector,3)  // Vector class with single precision
};

inline       TVectorF &TVectorF::Use           (Int_t n,Float_t *data) { return Use(0,n-1,data); }
inline       TVectorF &TVectorF::Use           (TVectorF &v)
                                                        { 
                                                          Assert(v.IsValid());
                                                          return Use(v.GetLwb(),v.GetUpb(),v.GetMatrixArray());
                                                        }
inline       TVectorF  TVectorF::GetSub        (Int_t row_lwb,Int_t row_upb,Option_t *option) const
                                                        { 
                                                          TVectorF tmp;
                                                          this->GetSub(row_lwb,row_upb,tmp,option);
                                                          return tmp;
                                                        }

inline const Float_t  &TVectorF::operator()(Int_t ind) const
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
Float_t   operator*     (const TVectorF    &source1,const TVectorF &source2);
TVectorF  operator*     (const TMatrixF    &a,      const TVectorF &source);
TVectorF  operator*     (const TMatrixFSym &a,      const TVectorF &source);
TVectorF  operator*     (      Float_t      val,    const TVectorF &source);
TVectorF  &Add          (      TVectorF    &target,       Float_t  scalar,const TVectorF &source);
TVectorF  &AddElemMult  (      TVectorF    &target,       Float_t  scalar,const TVectorF &source1,
                         const TVectorF    &source2);
TVectorF  &AddElemMult  (      TVectorF    &target,       Float_t  scalar,const TVectorF &source1,
                         const TVectorF    &source2,const TVectorF &select);
TVectorF  &AddElemDiv   (      TVectorF    &target,       Float_t  scalar,const TVectorF &source1,
                         const TVectorF    &source2);
TVectorF  &AddElemDiv   (      TVectorF    &target,       Float_t  scalar,const TVectorF &source1,
                         const TVectorF    &source2,const TVectorF &select);
TVectorF  &ElementMult  (      TVectorF    &target, const TVectorF &source);
TVectorF  &ElementMult  (      TVectorF    &target, const TVectorF &source,const TVectorF &select);
TVectorF  &ElementDiv   (      TVectorF    &target, const TVectorF &source);
TVectorF  &ElementDiv   (      TVectorF    &target, const TVectorF &source,const TVectorF &select);
Bool_t     AreCompatible(const TVectorF    &source1,const TVectorF &source2,Int_t verbose=0);
Bool_t     AreCompatible(const TVectorF    &source1,const TVectorD &source2,Int_t verbose=0);
void       Compare      (const TVectorF    &source1,const TVectorF &source2);

// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails

Bool_t VerifyVectorValue   (const TVectorF &m,Float_t val,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);
Bool_t VerifyVectorIdentity(const TVectorF &m1,const TVectorF &m2,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);
#endif
