// @(#)root/matrix:$Name:  $:$Id: TMatrixFBase.h,v 1.14 2004/06/21 15:53:12 brun Exp $
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

class TMatrixDBase;
class TElementActionF;
class TElementPosActionF;
class TMatrixFBase : public TObject {

private:
  Float_t *GetElements();  // This function is now obsolete (and is not implemented) you should use TMatrix::GetMatrixArray().

protected:
  Int_t     fNrows;               // number of rows
  Int_t     fNcols;               // number of columns
  Int_t     fRowLwb;              // lower bound of the row index
  Int_t     fColLwb;              // lower bound of the col index
  Int_t     fNelems;              // number of elements in matrix
  Int_t     fNrowIndex;           // length of row index array (= fNrows+1) wich is only used for sparse matrices

  Float_t   fTol;                 // sqrt(epsilon); epsilon is smallest number number so that  1+epsilon > 1
                                  //  fTol is used in matrix decomposition (like in inversion)

  enum {kSizeMax = 25};           // size data container on stack, see New_m(),Delete_m()
  enum {kWorkMax = 100};          // size of work array's in several routines

  Float_t   fDataStack[kSizeMax]; //! data container
  Bool_t    fIsOwner;             //!default kTRUE, when Use array kFALSE

  Float_t*  New_m   (Int_t size);
  void      Delete_m(Int_t size,Float_t*&);
  Int_t     Memcpy_m(Float_t *newp,const Float_t *oldp,Int_t copySize,
                     Int_t newSize,Int_t oldSize);

  virtual void Allocate (Int_t nrows,Int_t ncols,Int_t row_lwb = 0,
                         Int_t col_lwb = 0,Int_t init = 0,Int_t nr_nonzero = -1) = 0;

public:
  enum EMatrixStatusBits {
    kStatus = BIT(14) // set if matrix object is valid
  };
  enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
  enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult,kMultTranspose,kPlus,kMinus };

  TMatrixFBase() { fIsOwner = kTRUE;
                   fNelems = fNrowIndex = fNrows = fRowLwb = fNcols = fColLwb = 0; fTol = 0.; }

  virtual ~TMatrixFBase() {}

          inline       Int_t     GetRowLwb    () const { return fRowLwb; }
          inline       Int_t     GetRowUpb    () const { return fNrows+fRowLwb-1; }
          inline       Int_t     GetNrows     () const { return fNrows; }
          inline       Int_t     GetColLwb    () const { return fColLwb; }
          inline       Int_t     GetColUpb    () const { return fNcols+fColLwb-1; }
          inline       Int_t     GetNcols     () const { return fNcols; }
          inline       Int_t     GetNoElements() const { return fNelems; }
          inline       Float_t   GetTol       () const { return fTol; }

  virtual        const Float_t  *GetMatrixArray  () const = 0;
  virtual              Float_t  *GetMatrixArray  ()       = 0;
  virtual        const Int_t    *GetRowIndexArray() const = 0;
  virtual              Int_t    *GetRowIndexArray()       = 0;
  virtual        const Int_t    *GetColIndexArray() const = 0;
  virtual              Int_t    *GetColIndexArray()       = 0;

  virtual              TMatrixFBase &SetRowIndexArray(Int_t *data) = 0;
  virtual              TMatrixFBase &SetColIndexArray(Int_t *data) = 0;
  virtual              TMatrixFBase &SetMatrixArray  (const Float_t *data,Option_t *option="");
          inline       Float_t       SetTol          (Float_t tol);

  virtual void   Clear      (Option_t *option="") = 0;

  inline  void   Invalidate ()       { SetBit(kStatus); }
  inline  void   MakeValid  ()       { ResetBit(kStatus); }
  inline  Bool_t IsValid    () const { return !TestBit(kStatus); }
  inline  Bool_t IsOwner    () const { return fIsOwner; }
          Bool_t IsSymmetric() const;

  virtual TMatrixFBase &GetSub         (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                        TMatrixFBase &target,Option_t *option="S") const = 0;
  virtual TMatrixFBase &SetSub         (Int_t row_lwb,Int_t col_lwb,const TMatrixFBase &source) = 0;

  virtual void          GetMatrix2Array(Float_t *data,Option_t *option="") const;
  virtual TMatrixFBase &InsertRow      (Int_t row,Int_t col,const Float_t *v,Int_t n = -1);
  virtual void          ExtractRow     (Int_t row,Int_t col,      Float_t *v,Int_t n = -1) const;

  virtual TMatrixFBase &Shift          (Int_t row_shift,Int_t col_shift);
  virtual TMatrixFBase &ResizeTo       (Int_t nrows,Int_t ncols,Int_t nr_nonzeros=-1);
  virtual TMatrixFBase &ResizeTo       (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t nr_nonzeros=-1);
  inline  TMatrixFBase &ResizeTo       (const TMatrixFBase &m) {
                                         return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
                                       }

  virtual Double_t Determinant() const                          { AbstractMethod("Determinant()"); return 0.; }
  virtual void     Determinant(Double_t &d1,Double_t &d2) const { AbstractMethod("Determinant()"); d1 = 0.; d2 = 0.; }

  virtual TMatrixFBase &Zero       ();
  virtual TMatrixFBase &Abs        ();
  virtual TMatrixFBase &Sqr        ();
  virtual TMatrixFBase &Sqrt       ();
  virtual TMatrixFBase &UnitMatrix ();

  virtual TMatrixFBase &NormByDiag (const TVectorF &v,Option_t *option="D");

  virtual Float_t  RowNorm    () const;
  virtual Float_t  ColNorm    () const;
  virtual Float_t  E2Norm     () const;
  inline  Float_t  NormInf    () const { return RowNorm(); }
  inline  Float_t  Norm1      () const { return ColNorm(); }
  virtual Int_t    NonZeros   () const;
  virtual Float_t  Sum        () const;
  virtual Float_t  Min        () const;
  virtual Float_t  Max        () const;

  void Draw (Option_t *option="");       // *MENU*
  void Print(Option_t *name  ="") const; // *MENU*

  virtual Float_t  operator()(Int_t rown,Int_t coln) const = 0;
  virtual Float_t &operator()(Int_t rown,Int_t coln)       = 0;

  Bool_t operator==(Float_t val) const;
  Bool_t operator!=(Float_t val) const;
  Bool_t operator< (Float_t val) const;
  Bool_t operator<=(Float_t val) const;
  Bool_t operator> (Float_t val) const;
  Bool_t operator>=(Float_t val) const;

  virtual TMatrixFBase &Apply(const TElementActionF    &action);
  virtual TMatrixFBase &Apply(const TElementPosActionF &action);

  virtual TMatrixFBase &Randomize(Float_t alpha,Float_t beta,Double_t &seed);

  ClassDef(TMatrixFBase,4) // Dense Matrix base class (single precision)
};

Float_t TMatrixFBase::SetTol(Float_t newTol)
{
  const Float_t oldTol = fTol;
  if (newTol >= 0.0)
    fTol = newTol;
  return oldTol;
}

Bool_t  operator==   (const TMatrixFBase &m1,const TMatrixFBase &m2);
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

#endif
