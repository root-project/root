// @(#)root/matrix:$Name:  $:$Id: TMatrixDBase.h,v 1.12 2004/05/27 06:39:53 brun Exp $
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

#include <limits.h>
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

class TMatrixFBase;
class TElementActionD;
class TElementPosActionD;
class TMatrixDBase : public TObject {

private:
  Double_t *GetElements();  // This function is now obsolete (and is not implemented) you should use TMatrix::GetMatrixArray().

protected:
  Int_t     fNrows;               // number of rows
  Int_t     fNcols;               // number of columns
  Int_t     fRowLwb;              // lower bound of the row index
  Int_t     fColLwb;              // lower bound of the col index
  Int_t     fNelems;              // number of elements in matrix
  Int_t     fNrowIndex;           // length of row index array (= fNrows+1) wich is only used for sparse matrices

  Double_t  fTol;                 // sqrt(epsilon); epsilon is smallest number number so that  1+epsilon > 1
                                  //  fTol is used in matrix decomposition (like in inversion)

  enum {kSizeMax = 25};           // size data container on stack, see New_m(),Delete_m()
  enum {kWorkMax = 100};          // size of work array's in several routines

  Double_t  fDataStack[kSizeMax]; //! data container
  Bool_t    fIsOwner;             //!default kTRUE, when Use array kFALSE

  Double_t* New_m   (Int_t size);
  void      Delete_m(Int_t size,Double_t*&);
  Int_t     Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  static  void DoubleLexSort (Int_t n,Int_t *first,Int_t *second,Double_t *data);
  static  void IndexedLexSort(Int_t n,Int_t *first,Int_t swapFirst,
                              Int_t *second,Int_t swapSecond,Int_t *index);

  virtual void Allocate      (Int_t nrows,Int_t ncols,Int_t row_lwb = 0,
		              Int_t col_lwb = 0,Int_t init = 0,Int_t nr_nonzero = -1) = 0;

public:
  enum EMatrixStatusBits {
    kStatus = BIT(14) // set if matrix object is valid
  };

  enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
  enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult,kMultTranspose,kPlus,kMinus };

  TMatrixDBase() { SetBit(kStatus); fIsOwner = kTRUE; 
                   fNelems = fNrowIndex = fNrows = fRowLwb = fNcols = fColLwb = 0; fTol = 0.; }

  virtual ~TMatrixDBase() {}

          inline       Int_t     GetRowLwb     () const { return fRowLwb; }
          inline       Int_t     GetRowUpb     () const { return fNrows+fRowLwb-1; }
          inline       Int_t     GetNrows      () const { return fNrows; }
          inline       Int_t     GetColLwb     () const { return fColLwb; }
          inline       Int_t     GetColUpb     () const { return fNcols+fColLwb-1; }
          inline       Int_t     GetNcols      () const { return fNcols; }
          inline       Int_t     GetNoElements () const { return fNelems; }
          inline       Double_t  GetTol        () const { return fTol; }

  virtual        const Double_t *GetMatrixArray  () const = 0;
  virtual              Double_t *GetMatrixArray  ()       = 0;
  virtual        const Int_t    *GetRowIndexArray() const = 0;
  virtual              Int_t    *GetRowIndexArray()       = 0;
  virtual        const Int_t    *GetColIndexArray() const = 0;
  virtual              Int_t    *GetColIndexArray()       = 0;

  virtual              void      SetRowIndexArray(Int_t *data) = 0;
  virtual              void      SetColIndexArray(Int_t *data) = 0;
  virtual              void      SetMatrixArray  (const Double_t *data,Option_t *option="");
          inline       Double_t  SetTol        (Double_t tol);

  virtual void   Clear      (Option_t *option="") = 0;

  inline  void   Invalidate ()       { ResetBit(kStatus); }
  inline  void   MakeValid  ()       { SetBit(kStatus); }
  inline  Bool_t IsValid    () const { return TestBit(kStatus); }
  inline  Bool_t IsOwner    () const { return fIsOwner; }
  virtual Bool_t IsSymmetric() const;

  virtual void SetSub         (Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source) = 0;
  virtual void GetMatrix2Array(Double_t *data,Option_t *option="") const;
  virtual void InsertRow      (Int_t row,Int_t col,const Double_t *v,Int_t n = -1);
  virtual void ExtractRow     (Int_t row,Int_t col,      Double_t *v,Int_t n = -1) const;

  virtual void Shift   (Int_t row_shift,Int_t col_shift);
  virtual void ResizeTo(Int_t nrows,Int_t ncols,Int_t nr_nonzeros=-1);
  virtual void ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros=-1);
  inline  void ResizeTo(const TMatrixDBase &m) {
    ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
  }

  virtual Double_t Determinant() const                          { AbstractMethod("Determinant()"); return 0.; }
  virtual void     Determinant(Double_t &d1,Double_t &d2) const { AbstractMethod("Determinant()"); d1 = 0.; d2 = 0.; }

  virtual TMatrixDBase &Zero       ();
  virtual TMatrixDBase &Abs        ();
  virtual TMatrixDBase &Sqr        ();
  virtual TMatrixDBase &Sqrt       ();
  virtual TMatrixDBase &UnitMatrix ();

  virtual TMatrixDBase &NormByDiag (const TVectorD &v,Option_t *option="D");

  virtual Double_t RowNorm    () const;
  virtual Double_t ColNorm    () const;
  virtual Double_t E2Norm     () const;
  inline  Double_t NormInf    () const { return RowNorm(); }
  inline  Double_t Norm1      () const { return ColNorm(); }
  virtual Int_t    NonZeros   () const;
  virtual Double_t Sum        () const;
  virtual Double_t Min        () const;
  virtual Double_t Max        () const;

  void Draw (Option_t *option="");       // *MENU*
  void Print(Option_t *option="") const; // *MENU*

  virtual Double_t  operator()(Int_t rown,Int_t coln) const = 0;
  virtual Double_t &operator()(Int_t rown,Int_t coln)       = 0;

  Bool_t operator==(Double_t val) const;
  Bool_t operator!=(Double_t val) const;
  Bool_t operator< (Double_t val) const;
  Bool_t operator<=(Double_t val) const;
  Bool_t operator> (Double_t val) const;
  Bool_t operator>=(Double_t val) const;

  virtual TMatrixDBase &Apply(const TElementActionD    &action);
  virtual TMatrixDBase &Apply(const TElementPosActionD &action);

  virtual void Randomize(Double_t alpha,Double_t beta,Double_t &seed);

  ClassDef(TMatrixDBase,3) // Matrix base class (double precision)
};

Double_t TMatrixDBase::SetTol(Double_t newTol)
{
  const Double_t oldTol = fTol;
  if (newTol >= 0.0)
    fTol = newTol;
  return oldTol;
}

Bool_t   operator==   (const TMatrixDBase &m1,const TMatrixDBase &m2);
Double_t E2Norm       (const TMatrixDBase &m1,const TMatrixDBase &m2);
Bool_t   AreCompatible(const TMatrixDBase &m1,const TMatrixDBase &m2,Int_t verbose=0);
Bool_t   AreCompatible(const TMatrixDBase &m1,const TMatrixFBase &m2,Int_t verbose=0);
void     Compare      (const TMatrixDBase &m1,const TMatrixDBase &m2);

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
#ifndef ROOT_TMatrixDSparse
#include "TMatrixDSparse.h"
#endif
#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif

#endif
