// @(#)root/matrix:$Name:  $:$Id: TMatrixDBase.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDBase
#define ROOT_TMatrixDBase

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
#ifndef ROOT_TMatrixFBase
#include "TMatrixFBase.h"
#endif

class TMatrixDRow_const;
class TMatrixDRow;
class TMatrixFBase;
class TMatrixDBase : public TObject {

protected:
  Int_t     fNrows;               // number of rows
  Int_t     fNcols;               // number of columns
  Int_t     fNelems;              // number of elements in matrix
  Int_t     fRowLwb;              // lower bound of the row index
  Int_t     fColLwb;              // lower bound of the col index
  Double_t  fTol;                 // sqrt(epsilon); epsilon is smallest number number so that  1+epsilon > 1
                                  //  fTol is used in matrix decomposition (like in inversion)

  enum {kSizeMax = 25};           // size data container on stack, see New_m(),Delete_m()
  enum {kWorkMax = 100};          // size of work array's in several routines

  Double_t  fDataStack[kSizeMax]; //! data container
  Bool_t    fIsOwner;             //!default kTRUE, when Adopt array kFALSE

  Double_t* New_m   (Int_t size);
  void      Delete_m(Int_t size,Double_t*);
  Int_t     Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  virtual void Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0) = 0;

public:
  enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
  enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult };

  TMatrixDBase() { Invalidate(); }

  virtual ~TMatrixDBase() {}

          inline       Int_t     GetRowLwb    () const { return fRowLwb; }
          inline       Int_t     GetRowUpb    () const { return fNrows+fRowLwb-1; }
          inline       Int_t     GetNrows     () const { return fNrows; }
          inline       Int_t     GetColLwb    () const { return fColLwb; }
          inline       Int_t     GetColUpb    () const { return fNcols+fColLwb-1; }
          inline       Int_t     GetNcols     () const { return fNcols; }
          inline       Int_t     GetNoElements() const { return fNelems; }
          inline       Double_t  GetTol       () const { return fTol; }
  virtual        const Double_t *GetElements  () const = 0;
  virtual              Double_t *GetElements  ()       = 0;
          inline       Double_t  SetTol       (Double_t tol);

  virtual void Invalidate   ()       { fNrows = fNcols = fNelems = -1; }
  inline  Bool_t IsValid    () const { if (fNrows == -1) return kFALSE; return kTRUE; }
          Bool_t IsSymmetric() const;

  // Probably move this functionality to TMatrixDFlat
  virtual void GetMatrixElements(      Double_t *data, Option_t *option="") const;
  virtual void SetMatrixElements(const Double_t *data, Option_t *option="");

  virtual void Shift   (Int_t row_shift,Int_t col_shift);
  virtual void ResizeTo(Int_t nrows,Int_t ncols);
  virtual void ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  inline  void ResizeTo(const TMatrixDBase &m) {
    ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
  }

  virtual Double_t Determinant() const = 0;
  virtual void     Determinant(Double_t &d1,Double_t &d2) const =0;
  virtual Double_t RowNorm    () const;
  virtual Double_t ColNorm    () const;
  virtual Double_t E2Norm     () const;
  inline  Double_t NormInf    () const { return RowNorm(); }
  inline  Double_t Norm1      () const { return ColNorm(); }

  virtual void Clear(Option_t *option="") = 0;
  void Draw (Option_t *option="");       // *MENU*
  void Print(Option_t *option="") const; // *MENU*

  virtual const Double_t          &operator()(Int_t rown,Int_t coln) const = 0;
  virtual       Double_t          &operator()(Int_t rown,Int_t coln)       = 0;
          const TMatrixDRow_const  operator[](Int_t rown) const;
                TMatrixDRow        operator[](Int_t rown)      ;

  Bool_t operator==(Double_t val) const;
  Bool_t operator!=(Double_t val) const;
  Bool_t operator< (Double_t val) const;
  Bool_t operator<=(Double_t val) const;
  Bool_t operator> (Double_t val) const;
  Bool_t operator>=(Double_t val) const;

  friend Double_t E2Norm       (const TMatrixDBase &m1,const TMatrixDBase &m2);
  friend Bool_t   AreCompatible(const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose);
  friend void     Compare      (const TMatrixDBase &m1,const TMatrixDBase &m2);

  ClassDef(TMatrixDBase,2) // Matrix class (double precision)
};

Double_t TMatrixDBase::SetTol(Double_t newTol)
{
  const Double_t oldTol = fTol;
  if (newTol >= 0.0)
    fTol = newTol;
  return oldTol;
}

Bool_t AreCompatible(const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose=0);
Bool_t AreCompatible(const TMatrixDBase &m1,const TMatrixFBase &m2,Int_t verbose=0);
void   Compare      (const TMatrixDBase &m1,const TMatrixDBase &m2);

// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails

Bool_t VerifyMatrixValue   (const TMatrixDBase &m, Double_t val,
                            Int_t verbose=1,Double_t maxDevAllow=DBL_EPSILON);
Bool_t VerifyMatrixIdentity(const TMatrixDBase &m1,const TMatrixDBase &m2,
                            Int_t verbose=1,Double_t maxDevAllow=DBL_EPSILON);

#ifndef ROOT_TMatrixDUtils
#include "TMatrixDUtils.h"
#endif
#ifndef ROOT_TMatrixDLazy
#include "TMatrixDLazy.h"
#endif
#ifndef ROOT_TMatrixDSym
#include "TMatrixDSym.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif

inline const TMatrixDRow_const TMatrixDBase::operator[](Int_t rown) const { return TMatrixDRow_const(*this,rown); }
inline       TMatrixDRow       TMatrixDBase::operator[](Int_t rown)       { return TMatrixDRow      (*this,rown); }

#endif
