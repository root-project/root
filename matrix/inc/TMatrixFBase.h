// @(#)root/matrix:$Name:  $:$Id: TMatrixFBase.h,v 1.2 2004/01/25 23:28:44 rdm Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFBase
#define ROOT_TMatrixFBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix Base class                                                    //
//                                                                      //
//  matrix properties are stored here, however the data storage is part //
//  of the derived classes                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#ifndef ROOT_TClass
#include "TClass.h"
#endif
#ifndef ROOT_TPluginManager
#include "TPluginManager.h"
#endif
#ifndef ROOT_TVirtualUtilHist
#include "TVirtualUtilHist.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif

class TMatrixFRow_const;
class TMatrixFRow;
class TMatrixDBase;
class TMatrixFBase : public TObject {

protected:
  Int_t     fNrows;               // number of rows
  Int_t     fNcols;               // number of columns
  Int_t     fNelems;              // number of elements in matrix
  Int_t     fRowLwb;              // lower bound of the row index
  Int_t     fColLwb;              // lower bound of the col index
  Float_t   fTol;                 // sqrt(epsilon); epsilon is smallest number number so that  1+epsilon > 1
                                  //  fTol is used in matrix decomposition (like in inversion)

  enum {kSizeMax = 25};           // size data container on stack, see New_m(),Delete_m()
  enum {kWorkMax = 100};          // size of work array's in several routines

  Float_t   fDataStack[kSizeMax]; //! data container
  Bool_t    fIsOwner;             //!default kTRUE, when Adopt array kFALSE

  Float_t* New_m   (Int_t size);
  void      Delete_m(Int_t size,Float_t*);
  Int_t     Memcpy_m(Float_t *newp,const Float_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0) = 0;

public:
  enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
  enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult };

  TMatrixFBase() { Invalidate(); }

  virtual ~TMatrixFBase() {}

          inline       Int_t     GetRowLwb    () const { return fRowLwb; }
          inline       Int_t     GetRowUpb    () const { return fNrows+fRowLwb-1; }
          inline       Int_t     GetNrows     () const { return fNrows; }
          inline       Int_t     GetColLwb    () const { return fColLwb; }
          inline       Int_t     GetColUpb    () const { return fNcols+fColLwb-1; }
          inline       Int_t     GetNcols     () const { return fNcols; }
          inline       Int_t     GetNoElements() const { return fNelems; }
          inline       Float_t   GetTol       () const { return fTol; }
  virtual        const Float_t  *GetElements  () const = 0;
  virtual              Float_t  *GetElements  ()       = 0;
          inline       Float_t   SetTol       (Float_t tol);

  virtual void Invalidate   ()       { fNrows = fNcols = fNelems = -1; }
  inline  Bool_t IsValid    () const { if (fNrows == -1) return kFALSE; return kTRUE; }
          Bool_t IsSymmetric() const;

  // Probably move this functionality to TMatrixFFlat
  virtual void GetMatrixElements(      Float_t *data, Option_t *option="") const;
  virtual void SetMatrixElements(const Float_t *data, Option_t *option="");

  virtual void Shift   (Int_t row_shift,Int_t col_shift);
  virtual void ResizeTo(Int_t nrows,Int_t ncols);
  virtual void ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  inline  void ResizeTo(const TMatrixFBase &m) {
    ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
  }

  virtual Double_t Determinant() const = 0;
  virtual void     Determinant(Double_t &d1,Double_t &d2) const =0;
  virtual Float_t  RowNorm    () const;
  virtual Float_t  ColNorm    () const;
  virtual Float_t  E2Norm     () const;
  inline  Float_t  NormInf    () const { return RowNorm(); }
  inline  Float_t  Norm1      () const { return ColNorm(); }

  virtual void Clear(Option_t *option="") = 0;
  void Draw (Option_t *option="");       // *MENU*
  void Print(Option_t *option="") const; // *MENU*

  virtual const Float_t           &operator()(Int_t rown,Int_t coln) const = 0;
  virtual       Float_t           &operator()(Int_t rown,Int_t coln)       = 0;
          const TMatrixFRow_const  operator[](Int_t rown) const;
                TMatrixFRow        operator[](Int_t rown)      ;

  Bool_t operator==(Float_t val) const;
  Bool_t operator!=(Float_t val) const;
  Bool_t operator< (Float_t val) const;
  Bool_t operator<=(Float_t val) const;
  Bool_t operator> (Float_t val) const;
  Bool_t operator>=(Float_t val) const;

  ClassDef(TMatrixFBase,2) // Matrix class (double precision)
};

Float_t TMatrixFBase::SetTol(Float_t newTol)
{
  const Float_t oldTol = fTol;
  if (newTol >= 0.0)
    fTol = newTol;
  return oldTol;
}

Float_t E2Norm       (const TMatrixFBase &m1,const TMatrixFBase &m2);
Bool_t  AreCompatible(const TMatrixFBase &m1,const TMatrixFBase &m2,Int_t verbose=0);
Bool_t  AreCompatible(const TMatrixFBase &m1,const TMatrixDBase &m2,Int_t verbose=0);
void    Compare      (const TMatrixFBase &m1,const TMatrixFBase &m2);

// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails

Bool_t VerifyMatrixValue   (const TMatrixFBase &m, Float_t val,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);
Bool_t VerifyMatrixIdentity(const TMatrixFBase &m1,const TMatrixFBase &m2,
                            Int_t verbose=1,Float_t maxDevAllow=DBL_EPSILON);

#ifndef ROOT_TMatrixFUtils
#include "TMatrixFUtils.h"
#endif
#ifndef ROOT_TMatrixFLazy
#include "TMatrixFLazy.h"
#endif
#ifndef ROOT_TMatrixFSym
#include "TMatrixFSym.h"
#endif
#ifndef ROOT_TMatrixF
#include "TMatrixF.h"
#endif
#ifndef ROOT_TVectorF
#include "TVectorF.h"
#endif

inline const TMatrixFRow_const TMatrixFBase::operator[](Int_t rown) const { return TMatrixFRow_const(*this,rown); }
inline       TMatrixFRow       TMatrixFBase::operator[](Int_t rown)       { return TMatrixFRow      (*this,rown); }

#endif
