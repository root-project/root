// @(#)root/matrix:$Name:  $:$Id: TMatrixD.cxx,v 1.79 2005/03/30 07:19:46 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixD                                                             //
//                                                                      //
// Implementation of a general matrix in the linear algebra package     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TMatrixF.h"
#include "TMatrixDLazy.h"
#include "TMatrixDCramerInv.h"
#include "TDecompLU.h"
#include "TMatrixDEigen.h"

ClassImp(TMatrixD)

//______________________________________________________________________________
TMatrixD::TMatrixD(Int_t no_rows,Int_t no_cols)
{
  Allocate(no_rows,no_cols,0,0,1);
}

//______________________________________________________________________________
TMatrixD::TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1);
}

//______________________________________________________________________________
TMatrixD::TMatrixD(Int_t no_rows,Int_t no_cols,const Double_t *elements,Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]
  //
  // array elements are copied

  Allocate(no_rows,no_cols);
  SetMatrixArray(elements,option);
}

//______________________________________________________________________________
TMatrixD::TMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                   const Double_t *elements,Option_t *option)
{
  // array elements are copied

  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb);
  SetMatrixArray(elements,option);
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixD &another) : TMatrixDBase(another)
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixF &another)
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixDSym &another)
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixDSparse &another)
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixD::TMatrixD(EMatrixCreatorsOp1 op,const TMatrixD &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Example: TMatrixD a(10,12); ...; TMatrixD b(TMatrixDBase::kTransposed, a);
  // Supported operations are: kZero, kUnit, kTransposed, kInverted and kAtA.

  Assert(this != &prototype);
  Invalidate();

  Assert(prototype.IsValid());

  switch(op) {
    case kZero:
      Allocate(prototype.GetNrows(),prototype.GetNcols(),
               prototype.GetRowLwb(),prototype.GetColLwb(),1);
      break;

    case kUnit:
      Allocate(prototype.GetNrows(),prototype.GetNcols(),
               prototype.GetRowLwb(),prototype.GetColLwb(),1);
      UnitMatrix();
      break;

    case kTransposed:
      Allocate(prototype.GetNcols(), prototype.GetNrows(),
               prototype.GetColLwb(),prototype.GetRowLwb());
      Transpose(prototype);
      break;

    case kInverted:
    {
      Allocate(prototype.GetNrows(),prototype.GetNcols(),
               prototype.GetRowLwb(),prototype.GetColLwb(),1);
      *this = prototype;
      // Since the user can not control the tolerance of this newly created matrix
      // we put it to the smallest possible number 
      const Double_t oldTol = this->SetTol(DBL_MIN);
      this->Invert();
      this->SetTol(oldTol);
      break;
    }

    case kAtA:
      AtMultB(prototype,prototype);
      break;

    default:
      Error("TMatrixD(EMatrixCreatorOp1)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixD &a,EMatrixCreatorsOp2 op,const TMatrixD &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Example: TMatrixD a(10,12), b(12,5); ...; TMatrixD c(a, TMatrixDBase::kMult, b);
  // Supported operations are: kMult (a*b), kTransposeMult (a'*b), kInvMult (a^(-1)*b)

  Invalidate();

  Assert(a.IsValid());
  Assert(b.IsValid());

  switch(op) {
    case kMult:
      AMultB(a,b);
      break;

    case kTransposeMult:
      AtMultB(a,b);
      break;

    case kMultTranspose:
      AMultBt(a,b);
      break;

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      const Double_t oldTol = this->SetTol(DBL_MIN);
      this->Invert();
      this->SetTol(oldTol);
      *this *= b;
      break;
    }

    case kPlus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this += b;
      break;
    }

    case kMinus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this -= b;
      break;
    }

    default:
      Error("TMatrixD(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixD &a,EMatrixCreatorsOp2 op,const TMatrixDSym &b)
{
  Invalidate();

  Assert(a.IsValid());
  Assert(b.IsValid());

  switch(op) {
    case kMult:
      AMultB(a,b);
      break;

    case kTransposeMult:
      AtMultB(a,b);
      break;

    case kMultTranspose:
      AMultBt(a,b);
      break;

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      const Double_t oldTol = this->SetTol(DBL_MIN);
      this->Invert();
      this->SetTol(oldTol);
      *this *= b;
      break;
    }

    case kPlus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this += b;
      break; 
    }

    case kMinus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this -= b;
      break;
    }

    default:
      Error("TMatrixD(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixDSym &a,EMatrixCreatorsOp2 op,const TMatrixD &b)
{
  Invalidate();

  Assert(a.IsValid());
  Assert(b.IsValid());

  switch(op) {
    case kMult:
      AMultB(a,b);
      break;

    case kTransposeMult:
      AtMultB(a,b);
      break;

    case kMultTranspose:
      AMultBt(a,b);
      break;

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      const Double_t oldTol = this->SetTol(DBL_MIN);
      this->Invert();
      this->SetTol(oldTol);
      *this *= b;
      break;
    }

    case kPlus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this += b;
      break; 
    }

    case kMinus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this -= b;
      break;
    }

    default:
      Error("TMatrixD(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixDSym &a,EMatrixCreatorsOp2 op,const TMatrixDSym &b)
{
  Invalidate();

  Assert(a.IsValid());
  Assert(b.IsValid());

  switch(op) {
    case kMult:
      AMultB(a,b);
      break;

    case kTransposeMult:
      AtMultB(a,b);
      break;

    case kMultTranspose:
      AMultBt(a,b);
      break;

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      const Double_t oldTol = this->SetTol(DBL_MIN);
      this->Invert();
      this->SetTol(oldTol);
      *this *= b;
      break;
    }

    case kPlus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this += b;
      break; 
    }

    case kMinus:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      *this -= b;
      break;
    }

    default:
      Error("TMatrixD(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixD::TMatrixD(const TMatrixDLazy &lazy_constructor)
{
  Allocate(lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetColUpb()-lazy_constructor.GetColLwb()+1,
           lazy_constructor.GetRowLwb(),lazy_constructor.GetColLwb(),1);
  lazy_constructor.FillIn(*this);
}

//______________________________________________________________________________
void TMatrixD::Delete_m(Int_t size,Double_t *&m)
{ 
  // delete data pointer m, if it was assigned on the heap

  if (m) {
    if (size > kSizeMax)
      delete [] m;
    m = 0;
  }       
}

//______________________________________________________________________________
Double_t* TMatrixD::New_m(Int_t size)
{
  // return data pointer . if requested size <= kSizeMax, assign pointer
  // to the stack space

  if (size == 0) return 0;
  else {
    if ( size <= kSizeMax )
      return fDataStack;
    else {
      Double_t *heap = new Double_t[size];
      return heap;
    }
  }
}

//______________________________________________________________________________
Int_t TMatrixD::Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                         Int_t newSize,Int_t oldSize)
{
  // copy copySize doubles from *oldp to *newp . However take care of the
  // situation where both pointers are assigned to the same stack space

  if (copySize == 0 || oldp == newp)
    return 0;
  else {
    if ( newSize <= kSizeMax && oldSize <= kSizeMax ) {
      // both pointers are inside fDataStack, be careful with copy direction !
      if (newp > oldp) {
        for (Int_t i = copySize-1; i >= 0; i--)
          newp[i] = oldp[i];
      } else {
        for (Int_t i = 0; i < copySize; i++)
          newp[i] = oldp[i];
      }
    }
    else
      memcpy(newp,oldp,copySize*sizeof(Double_t));
  }
  return 0;
}

//______________________________________________________________________________
void TMatrixD::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,
                        Int_t init,Int_t /*nr_nonzeros*/)
{
  // Allocate new matrix. Arguments are number of rows, columns, row
  // lowerbound (0 default) and column lowerbound (0 default).

  if (no_rows < 0 || no_cols < 0)
  {
    Error("Allocate","no_rows=%d no_cols=%d",no_rows,no_cols);
    Invalidate();
    return;
  }

  MakeValid();
  fNrows   = no_rows;
  fNcols   = no_cols;
  fRowLwb  = row_lwb;
  fColLwb  = col_lwb;
  fNelems  = fNrows*fNcols;
  fIsOwner = kTRUE;
  fTol     = DBL_EPSILON;

  if (fNelems > 0) {
    fElements = New_m(fNelems);
    if (init)
      memset(fElements,0,fNelems*sizeof(Double_t));
  } else
    fElements = 0;
}

//______________________________________________________________________________
void TMatrixD::AMultB(const TMatrixD &a,const TMatrixD &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == &b) {
    Error("AMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Double_t *ap = a.GetMatrixArray();
  const Double_t *bp = b.GetMatrixArray();
        Double_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,fNrows,fNcols,a.GetNcols(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t na     = a.GetNoElements();
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = b.GetNcols();
  const Double_t * const ap = a.GetMatrixArray();
  const Double_t * const bp = b.GetMatrixArray();
        Double_t *       cp = this->GetMatrixArray();

  const Double_t *arp0 = ap;                     // Pointer to  A[i,0];
  while (arp0 < ap+na) {
    for (const Double_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of B, Start bcp = B[0,0]
      const Double_t *arp = arp0;                       // Pointer to the i-th row of A, reset to A[i,0]
      Double_t cij = 0;
      while (bcp < bp+nb) {                     // Scan the i-th row of A and
        cij += *arp++ * *bcp;                   // the j-th col of B
        bcp += ncolsb;
      }
      *cp++ = cij;
      bcp -= nb-1;                              // Set bcp to the (j+1)-th col
    }
    arp0 += ncolsa;                             // Set ap to the (i+1)-th row
  }

  Assert(cp == this->GetMatrixArray()+fNelems && arp0 == ap+na);
#endif
}

//______________________________________________________________________________
void TMatrixD::AMultB(const TMatrixDSym &a,const TMatrixD &b,Int_t constr)
{
  // Matrix multiplication, with A symmetric and B general.
  // Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&a)) {
    Error("AMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == &b) {
    Error("AMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

  const Double_t *ap1 = a.GetMatrixArray();
  const Double_t *bp1 = b.GetMatrixArray();
        Double_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dsymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,b.GetNcols(),0.0,cp1,fNcols);
#else
  const Double_t *bp2 = b.GetMatrixArray();
        Double_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t b_ij = *bp1++;
      *cp1 += b_ij*(*ap1);
      Double_t tmp = 0.0;
      const Double_t *ap2 = ap1+1;
      for (Int_t k = i+1; k < fNrows; k++) {
        const Int_t index_kj = k*fNcols+j;
        const Double_t a_ik = *ap2++;
        const Double_t b_kj = bp2[index_kj];
        cp2[index_kj] += a_ik*b_ij;
        tmp += a_ik*b_kj;
      }
      *cp1++ += tmp;
    }
    ap1 += fNrows+1;
  }  
#endif
}

//______________________________________________________________________________
void TMatrixD::AMultB(const TMatrixD &a,const TMatrixDSym &b,Int_t constr)
{
  // Matrix multiplication, with A general and B symmetric.
  // Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&b)) {
    Error("AMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

  const Double_t *ap1 = a.GetMatrixArray();
        Double_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  const Double_t *bp1 = b.GetMatrixArray();
  cblas_dsymm (CblasRowMajor,CblasRight,CblasUpper,fNrows,fNcols,1.0,
               bp1,b.GetNcols(),ap1,a.GetNcols(),0.0,cp1,fNcols);
#else
  for (Int_t i = 0; i < fNrows; i++) {
    const Double_t *bp1 = b.GetMatrixArray();
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t a_ij = *ap1++;
      *cp1 += a_ij*(*bp1);
      Double_t tmp = 0.0;
      const Double_t *ap2 = ap1;
      const Double_t *bp2 = bp1+1;
            Double_t *cp2 = cp1+1;
      for (Int_t k = j+1; k < fNcols; k++) {
        const Double_t a_ik = *ap2++;
        const Double_t b_jk = *bp2++;
        *cp2++ += a_ij*b_jk;
        tmp += a_ik*b_jk;
      }
      *cp1++ += tmp;
      bp1 += fNcols+1;
    }
  }
#endif
}

//______________________________________________________________________________
void TMatrixD::AMultB(const TMatrixDSym &a,const TMatrixDSym &b,Int_t constr)
{
  // Matrix multiplication, with A symmetric and B symmetric.
  // (Actually copied for the moment routine for B general)
  // Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&a)) {
    Error("AMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&b)) {
    Error("AMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

  const Double_t *ap1 = a.GetMatrixArray();
  const Double_t *bp1 = b.GetMatrixArray();
        Double_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dsymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,b.GetNcols(),0.0,cp1,fNcols);
#else
  const Double_t *bp2 = b.GetMatrixArray();
        Double_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t b_ij = *bp1++;
      *cp1 += b_ij*(*ap1);
      Double_t tmp = 0.0;
      const Double_t *ap2 = ap1+1;
      for (Int_t k = i+1; k < fNrows; k++) {
        const Int_t index_kj = k*fNcols+j;
        const Double_t a_ik = *ap2++;
        const Double_t b_kj = bp2[index_kj];
        cp2[index_kj] += a_ik*b_ij;
        tmp += a_ik*b_kj;
      }
      *cp1++ += tmp;
    }
    ap1 += fNrows+1;
  }
#endif
}

//______________________________________________________________________________
void TMatrixD::AtMultB(const TMatrixD &a,const TMatrixD &b,Int_t constr)
{
  // Create a matrix C such that C = A' * B. In other words,
  // c[i,j] = SUM{ a[k,i] * b[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNrows() != b.GetNrows() || a.GetRowLwb() != b.GetRowLwb()) {
    Error("AtMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AtMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == &b) {
    Error("AtMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNcols(),b.GetNcols(),a.GetColLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Double_t *ap = a.GetMatrixArray();
  const Double_t *bp = b.GetMatrixArray();
        Double_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = b.GetNcols();
  const Double_t * const ap = a.GetMatrixArray();
  const Double_t * const bp = b.GetMatrixArray();
        Double_t *       cp = this->GetMatrixArray();

  const Double_t *acp0 = ap;           // Pointer to  A[i,0];
  while (acp0 < ap+ncolsa) {
    for (const Double_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of B, Start bcp = B[0,0]
      const Double_t *acp = acp0;                       // Pointer to the i-th column of A, reset to A[0,i]
      Double_t cij = 0;
      while (bcp < bp+nb) {           // Scan the i-th column of A and
        cij += *acp * *bcp;           // the j-th col of B
        acp += ncolsa;
        bcp += ncolsb;
      }
      *cp++ = cij;
      bcp -= nb-1;                    // Set bcp to the (j+1)-th col
    }
    acp0++;                           // Set acp0 to the (i+1)-th col
  }

  Assert(cp == this->GetMatrixArray()+fNelems && acp0 == ap+ncolsa);
#endif
}

//______________________________________________________________________________
void TMatrixD::AtMultB(const TMatrixD &a,const TMatrixDSym &b,Int_t constr)
{
  // Create a matrix C such that C = A' * B. In other words,
  // c[i,j] = SUM{ a[k,i] * b[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNrows() != b.GetNrows() || a.GetRowLwb() != b.GetRowLwb()) {
    Error("AtMultB","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AtMultB","this == &a");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&b)) {
    Error("AtMultB","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNcols(),b.GetNcols(),a.GetColLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Double_t *ap = a.GetMatrixArray();
  const Double_t *bp = b.GetMatrixArray();
        Double_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  Double_t *cp1 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    const Double_t *ap1 = a.GetMatrixArray()+i; // i-column of a
    const Double_t *bp1 = b.GetMatrixArray();
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t a_ji = *ap1;
      *cp1 += a_ji*(*bp1);
      Double_t tmp = 0.0;
      const Double_t *ap2 = ap1+fNrows;
      const Double_t *bp2 = bp1+1;
            Double_t *cp2 = cp1+1;
      for (Int_t k = j+1; k < fNcols; k++) {
        const Double_t b_jk = *bp2++;
        *cp2++ += a_ji*b_jk;
        tmp += (*ap2) * b_jk;
        ap2 += fNrows;
      }
      *cp1++ += tmp;
      ap1 += fNrows;
      bp1 += fNcols+1;
    }
  }
#endif
}

//______________________________________________________________________________
void TMatrixD::AMultBt(const TMatrixD &a,const TMatrixD &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B^T.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMultBt","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == &a) {
    Error("AMultBt","this == &a");
    Invalidate();
    return;
  }

  if (this == &b) {
    Error("AMultBt","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNrows(),a.GetRowLwb(),b.GetRowLwb(),1);

#ifdef CBLAS
  const Double_t *ap = a.GetMatrixArray();
  const Double_t *bp = b.GetMatrixArray();
        Double_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasNoTrans,CblasTrans,fNrows,fNcols,a.GetNcols(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t na     = a.GetNoElements();
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = b.GetNcols();
  const Double_t * const ap = a.GetMatrixArray();
  const Double_t * const bp = b.GetMatrixArray();
        Double_t *       cp = this->GetMatrixArray();

  const Double_t *arp0 = ap;                    // Pointer to  A[i,0];
  while (arp0 < ap+na) {
    const Double_t *brp0 = bp;                  // Pointer to  B[j,0];
    while (brp0 < bp+nb) {
      const Double_t *arp = arp0;               // Pointer to the i-th row of A, reset to A[i,0]
      const Double_t *brp = brp0;               // Pointer to the j-th row of B, reset to B[j,0]
      Double_t cij = 0;
      while (brp < brp0+ncolsb)                 // Scan the i-th row of A and
        cij += *arp++ * *brp++;                 // the j-th row of B
      *cp++ = cij;
      brp0 += ncolsb;                           // Set brp0 to the (j+1)-th row
    }
    arp0 += ncolsa;                             // Set arp0 to the (i+1)-th row
  }

  Assert(cp == this->GetMatrixArray()+fNelems && arp0 == ap+na);
#endif
}

//______________________________________________________________________________
void TMatrixD::AMultBt(const TMatrixDSym &a,const TMatrixD &b,Int_t constr)
{
  // Matrix multiplication, with A symmetric and B general.
  // Create a matrix C such that C = A * B^T.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNcols() || a.GetColLwb() != b.GetColLwb()) {
    Error("AMultBt","A rows and B columns incompatible");
    Invalidate();
    return;
  }

  if (this == dynamic_cast<const TMatrixD *>(&a)) {
    Error("AMultBt","this == &a");
    Invalidate();
    return;
  }

  if (this == &b) {
    Error("AMultBt","this == &b");
    Invalidate();
    return;
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNrows(),a.GetRowLwb(),b.GetRowLwb(),1);

  const Double_t *ap1 = a.GetMatrixArray();
  const Double_t *bp1 = b.GetMatrixArray();
        Double_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dgemm (CblasRowMajor,CblasNoTrans,CblasTrans,fNrows,fNcols,a.GetNcols(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsb = b.GetNcols();
        Double_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t b_ji = *bp1;
      *cp1 += b_ji*(*ap1);
      Double_t tmp = 0.0;
      const Double_t *ap2 = ap1+1;
      const Double_t *bp2 = bp1+1;
      for (Int_t k = i+1; k < fNrows; k++) {
        const Int_t index_kj = k*fNcols+j;
        const Double_t a_ik = *ap2++;
        const Double_t b_jk = *bp2++;
        cp2[index_kj] += a_ik*b_ji;
        tmp += a_ik*b_jk;
      }
      *cp1++ += tmp;
      bp1 += ncolsb;
    }
    ap1 += fNrows+1;
    bp1 -= nb-1;
  }
#endif
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Use(Int_t row_lwb,Int_t row_upb,
                        Int_t col_lwb,Int_t col_upb,Double_t *data)
{
  if (row_upb < row_lwb)
  {
    Error("Use","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
    Invalidate();
    return *this;
  }
  if (col_upb < col_lwb)
  {
    Error("Use","col_upb=%d < col_lwb=%d",col_upb,col_lwb);
    Invalidate();
    return *this;
  }

  Clear();
  fNrows    = row_upb-row_lwb+1;
  fNcols    = col_upb-col_lwb+1;
  fRowLwb   = row_lwb;
  fColLwb   = col_lwb;
  fNelems   = fNrows*fNcols;
  fElements = data;
  fIsOwner  = kFALSE;

  return *this;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixD::GetSub(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                               TMatrixDBase &target,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][col_lwb..col_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..col_upb-col_lwb+1] (default)
  // else          : return [row_lwb..row_upb][col_lwb..col_upb]

  Assert(IsValid());
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    target.Invalidate();
    return target;
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("GetSub","col_lwb out of bounds");
    target.Invalidate();
    return target;
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    target.Invalidate();
    return target;
  }
  if (col_upb < fColLwb || col_upb > fColLwb+fNcols-1) {
    Error("GetSub","col_upb out of bounds");
    target.Invalidate();
    return target;
  }
  if (row_upb < row_lwb || col_upb < col_lwb) {
    Error("GetSub","row_upb < row_lwb || col_upb < col_lwb");
    target.Invalidate();
    return target;
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t shift = (opt.Contains("S")) ? 1 : 0;

  const Int_t row_lwb_sub = (shift) ? 0               : row_lwb;
  const Int_t row_upb_sub = (shift) ? row_upb-row_lwb : row_upb;
  const Int_t col_lwb_sub = (shift) ? 0               : col_lwb;
  const Int_t col_upb_sub = (shift) ? col_upb-col_lwb : col_upb;

  target.ResizeTo(row_lwb_sub,row_upb_sub,col_lwb_sub,col_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;
  const Int_t ncols_sub = col_upb_sub-col_lwb_sub+1;

  if (target.GetRowIndexArray() && target.GetColIndexArray()) {
    for (Int_t irow = 0; irow < nrows_sub; irow++) {
      for (Int_t icol = 0; icol < ncols_sub; icol++) {
        target(irow+row_lwb_sub,icol+col_lwb_sub) = (*this)(row_lwb+irow,col_lwb+icol);
      }
    }
  } else {
    const Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNcols+(col_lwb-fColLwb);
          Double_t *bp = target.GetMatrixArray();

    for (Int_t irow = 0; irow < nrows_sub; irow++) {
      const Double_t *ap_sub = ap;
      for (Int_t icol = 0; icol < ncols_sub; icol++) {
        *bp++ = *ap_sub++;
      }
      ap += fNcols;
    }
  }

  return target;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixD::SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixDBase &source)
{
  // Insert matrix source starting at [row_lwb][col_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source][col_lwb..col_lwb+ncols_source];
  
  Assert(IsValid());
  Assert(source.IsValid());
  
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("SetSub","row_lwb outof bounds");
    Invalidate();
    return *this;
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("SetSub","col_lwb outof bounds");
    Invalidate();
    return *this;
  }
  const Int_t nRows_source = source.GetNrows();
  const Int_t nCols_source = source.GetNcols();
  if (row_lwb+nRows_source > fRowLwb+fNrows || col_lwb+nCols_source > fColLwb+fNcols) {
    Error("SetSub","source matrix too large");
    Invalidate();
    return *this;
  }
  
  if (source.GetRowIndexArray() && source.GetColIndexArray()) {
    const Int_t rowlwb_s = source.GetRowLwb();
    const Int_t collwb_s = source.GetColLwb();
    for (Int_t irow = 0; irow < nRows_source; irow++) {
      for (Int_t icol = 0; icol < nCols_source; icol++) {
        (*this)(row_lwb+irow,col_lwb+icol) = source(rowlwb_s+irow,collwb_s+icol);
      }
    }
  } else {
    const Double_t *bp = source.GetMatrixArray();
          Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNcols+(col_lwb-fColLwb);
  
    for (Int_t irow = 0; irow < nRows_source; irow++) {
      Double_t *ap_sub = ap;
      for (Int_t icol = 0; icol < nCols_source; icol++) {
        *ap_sub++ = *bp++;
      }
      ap += fNcols;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixD::ResizeTo(Int_t nrows,Int_t ncols,Int_t /*nr_nonzeros*/)
{
  // Set size of the matrix to nrows x ncols
  // New dynamic elements are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  Assert(IsValid());
  if (!fIsOwner) {
    Error("ResizeTo(Int_t,Int_t)","Not owner of data array,cannot resize");
    Invalidate();
    return *this;
  }

  if (fNelems > 0) {
    if (fNrows == nrows && fNcols == ncols)
      return *this;
    else if (nrows == 0 || ncols == 0) {
      fNrows = nrows; fNcols = ncols;
      Clear();
      return *this;
    }

    Double_t    *elements_old = GetMatrixArray();
    const Int_t  nelems_old   = fNelems;
    const Int_t  nrows_old    = fNrows;
    const Int_t  ncols_old    = fNcols;

    Allocate(nrows,ncols);
    Assert(IsValid());

    Double_t *elements_new = GetMatrixArray();
    // new memory should be initialized but be careful ot to wipe out the stack
    // storage. Initialize all when old or new storage was on the heap
    if (fNelems > kSizeMax || nelems_old > kSizeMax)
      memset(elements_new,0,fNelems*sizeof(Double_t));
    else if (fNelems > nelems_old)
      memset(elements_new+nelems_old,0,(fNelems-nelems_old)*sizeof(Double_t));

    // Copy overlap
    const Int_t ncols_copy = TMath::Min(fNcols,ncols_old); 
    const Int_t nrows_copy = TMath::Min(fNrows,nrows_old); 

    const Int_t nelems_new = fNelems;
    if (ncols_old < fNcols) {
      for (Int_t i = nrows_copy-1; i >= 0; i--)
        Memcpy_m(elements_new+i*fNcols,elements_old+i*ncols_old,ncols_copy,
                 nelems_new,nelems_old);
    } else {
      for (Int_t i = 0; i < nrows_copy; i++)
        Memcpy_m(elements_new+i*fNcols,elements_old+i*ncols_old,ncols_copy,
                 nelems_new,nelems_old);
    }

    Delete_m(nelems_old,elements_old);
  } else {
    Allocate(nrows,ncols,0,0,1);
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDBase &TMatrixD::ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                 Int_t /*nr_nonzeros*/)
{
  // Set size of the matrix to [row_lwb:row_upb] x [col_lwb:col_upb]
  // New dynamic elemenst are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleted.

  Assert(IsValid());
  if (!fIsOwner) {
    Error("ResizeTo(Int_t,Int_t,Int_t,Int_t)","Not owner of data array,cannot resize");
    Invalidate();
    return *this;
  }

  const Int_t new_nrows = row_upb-row_lwb+1;
  const Int_t new_ncols = col_upb-col_lwb+1;

  if (fNelems > 0) {

    if (fNrows  == new_nrows  && fNcols  == new_ncols &&
        fRowLwb == row_lwb    && fColLwb == col_lwb)
       return *this;
    else if (new_nrows == 0 || new_ncols == 0) {
      fNrows = new_nrows; fNcols = new_ncols;
      fRowLwb = row_lwb; fColLwb = col_lwb;
      Clear();
      return *this;
    }

    Double_t    *elements_old = GetMatrixArray();
    const Int_t  nelems_old   = fNelems;
    const Int_t  nrows_old    = fNrows;
    const Int_t  ncols_old    = fNcols;
    const Int_t  rowLwb_old   = fRowLwb;
    const Int_t  colLwb_old   = fColLwb;

    Allocate(new_nrows,new_ncols,row_lwb,col_lwb);
    Assert(IsValid());

    Double_t *elements_new = GetMatrixArray();
    // new memory should be initialized but be careful ot to wipe out the stack
    // storage. Initialize all when old or new storag ewas on the heap
    if (fNelems > kSizeMax || nelems_old > kSizeMax)
      memset(elements_new,0,fNelems*sizeof(Double_t));
    else if (fNelems > nelems_old)
      memset(elements_new+nelems_old,0,(fNelems-nelems_old)*sizeof(Double_t));

    // Copy overlap
    const Int_t rowLwb_copy = TMath::Max(fRowLwb,rowLwb_old); 
    const Int_t colLwb_copy = TMath::Max(fColLwb,colLwb_old); 
    const Int_t rowUpb_copy = TMath::Min(fRowLwb+fNrows-1,rowLwb_old+nrows_old-1); 
    const Int_t colUpb_copy = TMath::Min(fColLwb+fNcols-1,colLwb_old+ncols_old-1); 

    const Int_t nrows_copy = rowUpb_copy-rowLwb_copy+1;
    const Int_t ncols_copy = colUpb_copy-colLwb_copy+1;

    if (nrows_copy > 0 && ncols_copy > 0) {
      const Int_t colOldOff = colLwb_copy-colLwb_old;
      const Int_t colNewOff = colLwb_copy-fColLwb;
      if (ncols_old < fNcols) {
        for (Int_t i = nrows_copy-1; i >= 0; i--) {
          const Int_t iRowOld = rowLwb_copy+i-rowLwb_old;
          const Int_t iRowNew = rowLwb_copy+i-fRowLwb;
          Memcpy_m(elements_new+iRowNew*fNcols+colNewOff,
                   elements_old+iRowOld*ncols_old+colOldOff,ncols_copy,fNelems,nelems_old);
        }
      } else {
        for (Int_t i = 0; i < nrows_copy; i++) {
          const Int_t iRowOld = rowLwb_copy+i-rowLwb_old;
          const Int_t iRowNew = rowLwb_copy+i-fRowLwb;
          Memcpy_m(elements_new+iRowNew*fNcols+colNewOff,
                   elements_old+iRowOld*ncols_old+colOldOff,ncols_copy,fNelems,nelems_old);
        }
      }
    }

    Delete_m(nelems_old,elements_old);
  } else {
    Allocate(new_nrows,new_ncols,row_lwb,col_lwb,1);
  }

  return *this;
}

//______________________________________________________________________________
Double_t TMatrixD::Determinant() const
{
  const TMatrixD &tmp = *this;
  TDecompLU lu(tmp,fTol);
  Double_t d1,d2;
  lu.Det(d1,d2);
  return d1*TMath::Power(2.0,d2);
}

//______________________________________________________________________________
void TMatrixD::Determinant(Double_t &d1,Double_t &d2) const
{
  const TMatrixD &tmp = *this;
  TDecompLU lu(tmp,fTol);
  lu.Det(d1,d2);
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Invert(Double_t *det)
{
  // Invert the matrix and calculate its determinant

  Assert(IsValid());
  TDecompLU::InvertLU(*this,fTol,det);

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::InvertFast(Double_t *det)
{
  // Invert the matrix and calculate its determinant

  Assert(IsValid());

  const Char_t nRows = Char_t(GetNrows());
  switch (nRows) {
    case 1:
    {
     if (GetNrows() != GetNcols() || GetRowLwb() != GetColLwb()) {
        Error("Invert()","matrix should be square");
        Invalidate();
      } else {
        Double_t *pM = this->GetMatrixArray();
        if (*pM == 0.) {
           Error("InvertFast","matrix is singular");
          Invalidate();
        }
        else
          *pM = 1.0/(*pM);
      }
      return *this;
    }
    case 2:
    {
      TMatrixDCramerInv::Inv2x2(*this,det);
      return *this;
    }
    case 3:
    {
      TMatrixDCramerInv::Inv3x3(*this,det);
      return *this;
    }
    case 4:
    {
      TMatrixDCramerInv::Inv4x4(*this,det);
      return *this;
    }
    case 5:
    {
      TMatrixDCramerInv::Inv5x5(*this,det);
      return *this;
    }
    case 6:
    {
      TMatrixDCramerInv::Inv6x6(*this,det);
      return *this;
    }

    default:
    {
      TDecompLU::InvertLU(*this,fTol,det);
      return *this;
    }
  }
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Transpose(const TMatrixD &source)
{
  // Transpose a matrix.
      
  Assert(IsValid());
  Assert(source.IsValid());
        
  if (this == &source) {
    Double_t *ap = this->GetMatrixArray();
    if (fNrows == fNcols && fRowLwb == fColLwb) {
      for (Int_t i = 0; i < fNrows; i++) {
        const Int_t off_i = i*fNrows;
        for (Int_t j = i+1; j < fNcols; j++) {
          const Int_t off_j = j*fNcols;
          const Double_t tmp = ap[off_i+j];
          ap[off_i+j] = ap[off_j+i];
          ap[off_j+i] = tmp;
        }
      }
    } else {
      Double_t *oldElems = new Double_t[source.GetNoElements()];
      memcpy(oldElems,source.GetMatrixArray(),source.GetNoElements()*sizeof(Double_t));
      const Int_t nrows_old  = fNrows;
      const Int_t ncols_old  = fNcols;
      const Int_t rowlwb_old = fRowLwb;
      const Int_t collwb_old = fColLwb;

      fNrows  = ncols_old;  fNcols  = nrows_old;
      fRowLwb = collwb_old; fColLwb = rowlwb_old;
      for (Int_t irow = fRowLwb; irow < fRowLwb+fNrows; irow++) {
        for (Int_t icol = fColLwb; icol < fColLwb+fNcols; icol++) {
          const Int_t off = (icol-collwb_old)*ncols_old;
          (*this)(irow,icol) = oldElems[off+irow-rowlwb_old];
        }
      }
      delete [] oldElems;
    }
  } else {
    if (fNrows  != source.GetNcols()  || fNcols  != source.GetNrows() ||
        fRowLwb != source.GetColLwb() || fColLwb != source.GetRowLwb())
    {
      Error("Transpose","matrix has wrong shape");
      Invalidate();
      return *this;
    }

    const Double_t *sp1 = source.GetMatrixArray();
    const Double_t *scp = sp1; // Row source pointer
          Double_t *tp  = this->GetMatrixArray();
    const Double_t * const tp_last = this->GetMatrixArray()+fNelems;

    // (This: target) matrix is traversed row-wise way,
    // whilst the source matrix is scanned column-wise
    while (tp < tp_last) {
      const Double_t *sp2 = scp++;

      // Move tp to the next elem in the row and sp to the next elem in the curr col
      while (sp2 < sp1+fNelems) {
        *tp++ = *sp2;
        sp2 += fNrows;
      }
    }
    Assert(tp == tp_last && scp == sp1+fNrows);
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Rank1Update(const TVectorD &v,Double_t alpha)
{
  // Perform a rank 1 operation on the matrix:
  //     A += alpha * v * v^T
      
  Assert(IsValid());
  Assert(v.IsValid());

  if (v.GetNoElements() < TMath::Max(fNrows,fNcols)) {
    Error("Rank1Update","vector too short");
    Invalidate();
    return *this;
  }

  const Double_t * const pv = v.GetMatrixArray();
        Double_t *mp = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    const Double_t tmp = alpha*pv[i];
    for (Int_t j = 0; j < fNcols; j++)
      *mp++ += tmp*pv[j];
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::Rank1Update(const TVectorD &v1,const TVectorD &v2,Double_t alpha)       
{
  // Perform a rank 1 operation on the matrix:                          
  //     A += alpha * v1 * v2^T

  Assert(IsValid());
  Assert(v1.IsValid());
  Assert(v2.IsValid());

  if (v1.GetNoElements() < fNrows) {
    Error("Rank1Update","vector v1 too short");
    Invalidate();
    return *this;
  }

  if (v2.GetNoElements() < fNcols) {
    Error("Rank1Update","vector v2 too short");
    Invalidate();
    return *this;
  }

  const Double_t * const pv1 = v1.GetMatrixArray();
  const Double_t * const pv2 = v2.GetMatrixArray();
        Double_t *mp = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    const Double_t tmp = alpha*pv1[i];
    for (Int_t j = 0; j < fNcols; j++)
      *mp++ += tmp*pv2[j];
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::NormByColumn(const TVectorD &v,Option_t *option)
{
  // Multiply/divide matrix columns by a vector:
  // option:
  // "D"   :  b(i,j) = a(i,j)/v(i)   i = 0,fNrows-1 (default)
  // else  :  b(i,j) = a(i,j)*v(i)

  Assert(IsValid());
  Assert(v.IsValid());

  if (v.GetNoElements() < fNrows) {
    Error("NormByColumn","vector shorter than matrix column");
    Invalidate();
    return *this;
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t divide = (opt.Contains("D")) ? 1 : 0;

  const Double_t* pv = v.GetMatrixArray();
        Double_t *mp = this->GetMatrixArray();
  const Double_t * const mp_last = mp+fNelems;

  if (divide) {
    for ( ; mp < mp_last; pv++) {
      for (Int_t j = 0; j < fNcols; j++)
      {
        Assert(*pv != 0.0);
        *mp++ /= *pv;
      }
    }
  } else {
    for ( ; mp < mp_last; pv++)
      for (Int_t j = 0; j < fNcols; j++)
        *mp++ *= *pv;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::NormByRow(const TVectorD &v,Option_t *option)
{
  // Multiply/divide matrix rows with a vector:
  // option:
  // "D"   :  b(i,j) = a(i,j)/v(j)   i = 0,fNcols-1 (default)
  // else  :  b(i,j) = a(i,j)*v(j)

  Assert(IsValid());
  Assert(v.IsValid());
  if (v.GetNoElements() < fNcols) {
    Error("NormByRow","vector shorter than matrix column");
    Invalidate();
    return *this;
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t divide = (opt.Contains("D")) ? 1 : 0;

  const Double_t *pv0 = v.GetMatrixArray();
  const Double_t *pv  = pv0;
        Double_t *mp  = this->GetMatrixArray();
  const Double_t * const mp_last = mp+fNelems;

  if (divide) {
    for ( ; mp < mp_last; pv = pv0 )
      for (Int_t j = 0; j < fNcols; j++) {
        Assert(*pv != 0.0);
        *mp++ /= *pv++;
      }
  } else {
    for ( ; mp < mp_last; pv = pv0 )
      for (Int_t j = 0; j < fNcols; j++)
        *mp++ *= *pv++;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(const TMatrixD &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixD &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNelems*sizeof(Double_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(const TMatrixF &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixF &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (dynamic_cast<TMatrixF *>(this) != &source) {
    TObject::operator=(source);
    const Float_t  * const ps = source.GetMatrixArray();
          Double_t * const pt = GetMatrixArray();
    for (Int_t i = 0; i < fNelems; i++)
      pt[i] = (Double_t) ps[i];
    fTol = (Double_t)source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(const TMatrixDSym &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixDSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if ((TMatrixDBase *)this != (TMatrixDBase *)&source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNelems*sizeof(Double_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(const TMatrixDSparse &source)
{
  if (GetNrows()  != source.GetNrows()  || GetNcols()  != source.GetNcols() ||
      GetRowLwb() != source.GetRowLwb() || GetColLwb() != source.GetColLwb()) {
    Error("operator=(const TMatrixDSparse &","matrices not compatible");
    Invalidate();
    return *this;
  }


  if ((TMatrixDBase *)this != (TMatrixDBase *)&source) {
    TObject::operator=(source);
    memset(fElements,0,fNelems*sizeof(Double_t));

    const Double_t * const sp = source.GetMatrixArray();
          Double_t *       tp = this->GetMatrixArray();

    const Int_t * const pRowIndex = source.GetRowIndexArray();
    const Int_t * const pColIndex = source.GetColIndexArray();

    for (Int_t irow = 0; irow < fNrows; irow++ ) {
      const Int_t off = irow*fNcols;
      const Int_t sIndex = pRowIndex[irow];
      const Int_t eIndex = pRowIndex[irow+1];
      for (Int_t index = sIndex; index < eIndex; index++)
        tp[off+pColIndex[index]] = sp[index];
    }
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(const TMatrixDLazy &lazy_constructor)
{
  Assert(IsValid());

  if (lazy_constructor.GetRowUpb() != GetRowUpb() ||
      lazy_constructor.GetColUpb() != GetColUpb() ||
      lazy_constructor.GetRowLwb() != GetRowLwb() ||
      lazy_constructor.GetColLwb() != GetColLwb()) {
    Error("operator=(const TMatrixDLazy&)", "matrix is incompatible with "
          "the assigned Lazy matrix");
    Invalidate();
    return *this;
  }

  lazy_constructor.FillIn(*this);
  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator=(Double_t val)
{
  // Assign val to every element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator+=(Double_t val)
{
  // Add val to every element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator-=(Double_t val)
{
  // Subtract val from every element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(Double_t val)
{
  // Multiply every element of the matrix with val.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator+=(const TMatrixD &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TMatrixD &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp = source.GetMatrixArray();
  Double_t *tp = this->GetMatrixArray();
  const Double_t * const tp_last = tp+fNelems;
  while (tp < tp_last)
    *tp++ += *sp++;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator+=(const TMatrixDSym &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TMatrixDSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp  = source.GetMatrixArray();
        Double_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < fNrows; i++) {
    sp  += i;
    trp += i;         // point to [i,i]
    tcp += i*fNcols;  // point to [i,i]
    for (Int_t j = i; j < fNcols; j++) {
      if (j > i) *tcp += *sp;
      *trp++ += *sp++;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator-=(const TMatrixD &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator=-(const TMatrixD &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp = source.GetMatrixArray();
  Double_t *tp = this->GetMatrixArray();
  const Double_t * const tp_last = tp+fNelems;
  while (tp < tp_last)
    *tp++ -= *sp++;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator-=(const TMatrixDSym &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator=-(const TMatrixDSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp = source.GetMatrixArray();
        Double_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < fNrows; i++) {
    sp  += i;
    trp += i;         // point to [i,i]
    tcp += i*fNcols;  // point to [i,i]
    for (Int_t j = i; j < fNcols; j++) {
      if (j > i) *tcp -= *sp;
      *trp++ -= *sp++;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixD &source)
{
  // Compute target = target * source inplace. Strictly speaking, it can't be
  // done inplace, though only the row of the target matrix needs to be saved.
  // "Inplace" multiplication is only allowed when the 'source' matrix is square.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNcols != source.GetNrows() || fColLwb != source.GetRowLwb() ||
      fNcols != source.GetNcols() || fColLwb != source.GetColLwb()) {
    Error("operator*=(const TMatrixD &)","source matrix has wrong shape");
    Invalidate();
    return *this;
  }

  // Check for A *= A;
  const Double_t *sp;
  TMatrixD tmp;
  if (this == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  // One row of the old_target matrix
  Double_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Double_t *trp = work;
  if (fNcols > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Double_t[fNcols];
  }

        Double_t *cp   = this->GetMatrixArray();
  const Double_t *trp0 = cp; // Pointer to  target[i,0];
  const Double_t * const trp0_last = trp0+fNelems;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcols*sizeof(Double_t));        // copy the i-th row of target, Start at target[i,0]
    for (const Double_t *scp = sp; scp < sp+fNcols; ) {  // Pointer to the j-th column of source,
                                                         // Start scp = source[0,0]
      Double_t cij = 0;
      for (Int_t j = 0; j < fNcols; j++) {
        cij += trp[j] * *scp;                        // the j-th col of source
        scp += fNcols;
      }
      *cp++ = cij;
      scp -= source.GetNoElements()-1;               // Set bcp to the (j+1)-th col
    }
    trp0 += fNcols;                                  // Set trp0 to the (i+1)-th row
    Assert(trp0 == cp);
  }                                             

  Assert(cp == trp0_last && trp0 == trp0_last);
  if (isAllocated)
    delete [] trp;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixDSym &source)
{
  // Compute target = target * source inplace. Strictly speaking, it can't be
  // done inplace, though only the row of the target matrix needs to be saved.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNcols != source.GetNrows() || fColLwb != source.GetRowLwb()) {
    Error("operator*=(const TMatrixDSym &)","source matrix has wrong shape");
    Invalidate();
    return *this;
  }

  // Check for A *= A;
  const Double_t *sp;
  TMatrixD tmp;
  if ((TMatrixDBase *)this == (TMatrixDBase *)&source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  // One row of the old_target matrix
  Double_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Double_t *trp = work;
  if (fNcols > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Double_t[fNcols];
  }

        Double_t *cp   = this->GetMatrixArray();
  const Double_t *trp0 = cp; // Pointer to  target[i,0];
  const Double_t * const trp0_last = trp0+fNelems;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcols*sizeof(Double_t));        // copy the i-th row of target, Start at target[i,0]
    for (const Double_t *scp = sp; scp < sp+fNcols; ) {   // Pointer to the j-th column of source, 
                                                                     //Start scp = source[0,0]
      Double_t cij = 0;
      for (Int_t j = 0; j < fNcols; j++) {
        cij += trp[j] * *scp;                        // the j-th col of source
        scp += fNcols;
      }
      *cp++ = cij;
      scp -= source.GetNoElements()-1;               // Set bcp to the (j+1)-th col
    }
    trp0 += fNcols;                                  // Set trp0 to the (i+1)-th row
    Assert(trp0 == cp);
  }                                             

  Assert(cp == trp0_last && trp0 == trp0_last);
  if (isAllocated)
    delete [] trp;

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixDDiag_const &diag)
{
  // Multiply a matrix row by the diagonal of another matrix
  // matrix(i,j) *= diag(j), j=1,fNcols

  Assert(IsValid());
  Assert(diag.GetMatrix()->IsValid());
  Assert(fNcols == diag.GetNdiags());

  if (fNcols != diag.GetNdiags()) {
    Error("operator*=(const TMatrixDDiag_const &)","wrong diagonal length");
    Invalidate();
    return *this;
  }

  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Int_t inc = diag.GetInc();
  while (mp < mp_last) {
    const Double_t *dp = diag.GetPtr();
    for (Int_t j = 0; j < fNcols; j++) {
      *mp++ *= *dp;
      dp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator/=(const TMatrixDDiag_const &diag)
{
  // Divide a matrix row by the diagonal of another matrix
  // matrix(i,j) /= diag(j)

  Assert(IsValid());
  Assert(diag.GetMatrix()->IsValid());

  if (fNcols != diag.GetNdiags()) {
    Error("operator/=(const TMatrixDDiag_const &)","wrong diagonal length");
    Invalidate();
    return *this;
  }

  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Int_t inc = diag.GetInc();
  while (mp < mp_last) {
    const Double_t *dp = diag.GetPtr();
    for (Int_t j = 0; j < fNcols; j++) {
      Assert(*dp != 0.0);
      *mp++ /= *dp;
      dp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixDColumn_const &col)
{
  // Multiply a matrix by the column of another matrix
  // matrix(i,j) *= another(i,k) for fixed k

  const TMatrixDBase *mt = col.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNrows != mt->GetNrows()) {
    Error("operator*=(const TMatrixDColumn_const &)","wrong column length");
    Invalidate();
    return *this;
  }

  const Double_t * const endp = col.GetPtr()+mt->GetNoElements();
  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Double_t *cp = col.GetPtr();      //  ptr
  const Int_t inc = col.GetInc();
  while (mp < mp_last) {
    Assert(cp < endp);
    for (Int_t j = 0; j < fNcols; j++)
      *mp++ *= *cp;
    cp += inc;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator/=(const TMatrixDColumn_const &col)
{
  // Divide a matrix by the column of another matrix
  // matrix(i,j) /= another(i,k) for fixed k

  const TMatrixDBase *mt = col.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNrows != mt->GetNrows()) {
    Error("operator/=(const TMatrixDColumn_const &)","wrong column matrix");
    Invalidate();
    return *this;
  }

  const Double_t * const endp = col.GetPtr()+mt->GetNoElements();
  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Double_t *cp = col.GetPtr();      //  ptr
  const Int_t inc = col.GetInc();
  while (mp < mp_last) {
    Assert(cp < endp);
    Assert(*cp != 0.0);
    for (Int_t j = 0; j < fNcols; j++)
      *mp++ /= *cp;
    cp += inc;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator*=(const TMatrixDRow_const &row)
{
  // Multiply a matrix by the row of another matrix
  // matrix(i,j) *= another(k,j) for fixed k

  const TMatrixDBase *mt = row.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNcols != mt->GetNcols()) {
    Error("operator*=(const TMatrixDRow_const &)","wrong row length");
    Invalidate();
    return *this;
  }

  const Double_t * const endp = row.GetPtr()+mt->GetNoElements();
  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Int_t inc = row.GetInc();
  while (mp < mp_last) {
    const Double_t *rp = row.GetPtr();    // Row ptr
    for (Int_t j = 0; j < fNcols; j++) {
      Assert(rp < endp);
      *mp++ *= *rp;
      rp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixD &TMatrixD::operator/=(const TMatrixDRow_const &row)
{
  // Divide a matrix by the row of another matrix
  // matrix(i,j) /= another(k,j) for fixed k

  const TMatrixDBase *mt = row.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNcols != mt->GetNcols()) {
    Error("operator/=(const TMatrixDRow_const &)","wrong row length");
    Invalidate();
    return *this;
  }

  const Double_t * const endp = row.GetPtr()+mt->GetNoElements();
  Double_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Double_t * const mp_last = mp+fNelems;
  const Int_t inc = row.GetInc();
  while (mp < mp_last) {
    const Double_t *rp = row.GetPtr();    // Row ptr
    for (Int_t j = 0; j < fNcols; j++) {
      Assert(rp < endp);
      Assert(*rp != 0.0);
      *mp++ /= *rp;
      rp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
const TMatrixD TMatrixD::EigenVectors(TVectorD &eigenValues) const
{
  // Return a matrix containing the eigen-vectors ordered by descending values
  // of Re^2+Im^2 of the complex eigen-values .
  // If the matrix is asymmetric, only the real part of the eigen-values is
  // returned . For full functionality use TMatrixDEigen .

  if (!IsSymmetric())
    Warning("EigenVectors(TVectorD &)","Only real part of eigen-values will be returned");
  TMatrixDEigen eigen(*this);
  eigenValues.ResizeTo(fNrows);
  eigenValues = eigen.GetEigenValuesRe();
  return eigen.GetEigenVectors();
}

//______________________________________________________________________________
TMatrixD operator+(const TMatrixD &source1,const TMatrixD &source2)
{
  TMatrixD target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixD operator+(const TMatrixD &source1,const TMatrixDSym &source2)
{
  TMatrixD target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixD operator+(const TMatrixDSym &source1,const TMatrixD &source2)
{
  return operator+(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator+(const TMatrixD &source,Double_t val)
{
  TMatrixD target(source);
  target += val;
  return target;
}

//______________________________________________________________________________
TMatrixD operator+(Double_t val,const TMatrixD &source)
{
  return operator+(source,val);
}

//______________________________________________________________________________
TMatrixD operator-(const TMatrixD &source1,const TMatrixD &source2)
{
  TMatrixD target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixD operator-(const TMatrixD &source1,const TMatrixDSym &source2)
{
  TMatrixD target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixD operator-(const TMatrixDSym &source1,const TMatrixD &source2)
{
  return -1.*(operator-(source2,source1));
}

//______________________________________________________________________________
TMatrixD operator-(const TMatrixD &source,Double_t val)
{
  TMatrixD target(source);
  target -= val;
  return target;
}

//______________________________________________________________________________
TMatrixD operator-(Double_t val,const TMatrixD &source)
{
  return -1.0*operator-(source,val);
}

//______________________________________________________________________________
TMatrixD operator*(Double_t val,const TMatrixD &source)
{
  TMatrixD target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixD operator*(const TMatrixD &source,Double_t val)
{
  return operator*(val,source);
}

//______________________________________________________________________________
TMatrixD operator*(const TMatrixD &source1,const TMatrixD &source2)
{
  TMatrixD target(source1,TMatrixD::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixD operator*(const TMatrixD &source1,const TMatrixDSym &source2)
{
  TMatrixD target(source1,TMatrixD::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixD operator*(const TMatrixDSym &source1,const TMatrixD &source2)
{
  TMatrixD target(source1,TMatrixD::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixD operator*(const TMatrixDSym &source1,const TMatrixDSym &source2)
{
  TMatrixD target(source1,TMatrixD::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixD operator&&(const TMatrixD &source1,const TMatrixD &source2)
{
  // Logical AND

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator&&(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last)
    *tp++ = (*sp1++ != 0.0 && *sp2++ != 0.0);

  return target;
}

//______________________________________________________________________________
TMatrixD operator&&(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // Logical AND

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator&&(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j > i) *tcp = (*scp1 != 0.0) & (*sp2 != 0.0);
      *trp++ = (*srp1++ != 0.0 && *sp2++ != 0.0);
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator&&(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // Logical AND
  return operator&&(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator||(const TMatrixD &source1,const TMatrixD &source2)
{
  // Logical OR

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator||(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last)
    *tp++ = (*sp1++ != 0.0 || *sp2++ != 0.0);

  return target;
}

//______________________________________________________________________________
TMatrixD operator||(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // Logical OR

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator||(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j > i) *tcp = (*scp1 != 0.0) || (*sp2 != 0.0);
      *trp++ = (*srp1++ != 0.0 || *sp2++ != 0.0);
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator||(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // Logical OR
  return operator||(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator>(const TMatrixD &source1,const TMatrixD &source2)
{
  // source1 > source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator|(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last) {
    *tp++ = (*sp1) > (*sp2); sp1++; sp2++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator>(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // source1 > source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator>(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j > i) *tcp = (*scp1) > (*sp2);
      *trp++ = (*srp1) > (*sp2); srp1++; sp2++;
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator>(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // source1 > source2
  return operator<=(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator>=(const TMatrixD &source1,const TMatrixD &source2)
{
  // source1 >= source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator>=(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last) {
    *tp++ = (*sp1) >= (*sp2); sp1++; sp2++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator>=(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // source1 >= source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator>=(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j > i) *tcp = (*scp1) >= (*sp2);
      *trp++ = (*srp1) >= (*sp2); srp1++; sp2++;
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator>=(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // source1 >= source2
  return operator<(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator<=(const TMatrixD &source1,const TMatrixD &source2)
{
  // source1 <= source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator<=(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last) {
    *tp++ = (*sp1) <= (*sp2); sp1++; sp2++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator<=(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // source1 <= source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator<=(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j < i) *tcp = (*scp1) <= (*sp2);
      *trp++ = (*srp1) <= (*sp2); srp1++; sp2++;
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator<=(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // source1 <= source2
  return operator>(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator<(const TMatrixD &source1,const TMatrixD &source2)
{
  // source1 < source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator<(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp < tp_last) {
    *tp++ = (*sp1) < (*sp2); sp1++; sp2++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator<(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // source1 < source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator<(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;                // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j < i) *tcp = (*scp1) < (*sp2);
      *trp++ = (*srp1) < (*sp2); srp1++; sp2++;
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator<(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // source1 < source2
  return operator>=(source2,source1);
}

//______________________________________________________________________________
TMatrixD operator!=(const TMatrixD &source1,const TMatrixD &source2)
{
  // source1 != source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator!=(const TMatrixD&,const TMatrixD&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp1 = source1.GetMatrixArray();
  const Double_t *sp2 = source2.GetMatrixArray();
  Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp != tp_last) {
    *tp++ = (*sp1) != (*sp2); sp1++; sp2++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator!=(const TMatrixD &source1,const TMatrixDSym &source2)
{
  // source1 != source2

  TMatrixD target;

  if (!AreCompatible(source1,source2)) {
    Error("operator!=(const TMatrixD&,const TMatrixDSym&)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  target.ResizeTo(source1);

  const Int_t nelems = target.GetNoElements();
  const Int_t nrows  = target.GetNrows();
  const Int_t ncols  = target.GetNcols();

  const Double_t *srp1 = source1.GetMatrixArray();
  const Double_t *scp1 = srp1;
  const Double_t *sp2  = source2.GetMatrixArray();
        Double_t *trp  = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp  = trp;                     // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i != nrows; i++) {
    sp2 += i;
    srp1 += i; trp += i;              // point to [i,i]
    scp1 += i*ncols; tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j != ncols; j++) {
      if (j != i) *tcp = (*scp1) != (*sp2);
      *trp++ = (*srp1) != (*sp2); srp1++; sp2++;
      scp1 += ncols;
      tcp  += ncols;
    }
    scp1 -= nelems-1; // point to [0,i]
    tcp  -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator!=(const TMatrixDSym &source1,const TMatrixD &source2)
{
  // source1 != source2
  return operator!=(source2,source1);
}

/*
//______________________________________________________________________________
TMatrixD operator!=(const TMatrixD &source1,Double_t val)
{
  // source1 != val

  TMatrixD target; target.ResizeTo(source1);

  const Double_t *sp = source1.GetMatrixArray();
        Double_t *tp = target.GetMatrixArray();
  const Double_t * const tp_last = tp+target.GetNoElements();
  while (tp != tp_last) {
    *tp++ = (*sp != val); sp++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD operator!=(Double_t val,const TMatrixD &source1)
{
  // val != source1
  return operator!=(source1,val);
}
*/

//______________________________________________________________________________
TMatrixD &Add(TMatrixD &target,Double_t scalar,const TMatrixD &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    ::Error("Add(TMatrixD &,Double_t,const TMatrixD &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp  = source.GetMatrixArray();
        Double_t *tp  = target.GetMatrixArray();
  const Double_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp )
    *tp++ += scalar * (*sp++);

  return target;
}

//______________________________________________________________________________
TMatrixD &Add(TMatrixD &target,Double_t scalar,const TMatrixDSym &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    ::Error("Add(TMatrixD &,Double_t,const TMatrixDSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Double_t *sp  = source.GetMatrixArray();
        Double_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp  += i;
    trp += i;        // point to [i,i]
    tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      const Double_t tmp = scalar * *sp++;
      if (j > i) *tcp += tmp;
      tcp += ncols;
      *trp++ += tmp;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD &ElementMult(TMatrixD &target,const TMatrixD &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult(TMatrixD &,const TMatrixD &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp  = source.GetMatrixArray();
        Double_t *tp  = target.GetMatrixArray();
  const Double_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp )
    *tp++ *= *sp++;

  return target;
}

//______________________________________________________________________________
TMatrixD &ElementMult(TMatrixD &target,const TMatrixDSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult(TMatrixD &,const TMatrixDSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Double_t *sp  = source.GetMatrixArray();
        Double_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp  += i;
    trp += i;        // point to [i,i]
    tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      if (j > i) *tcp *= *sp;
      *trp++ *= *sp++;
      tcp += ncols;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixD &ElementDiv(TMatrixD &target,const TMatrixD &source)
{
  // Divide target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv(TMatrixD &,const TMatrixD &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Double_t *sp  = source.GetMatrixArray();
        Double_t *tp  = target.GetMatrixArray();
  const Double_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp ) {
    Assert(*sp != 0.0);
    *tp++ /= *sp++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixD &ElementDiv(TMatrixD &target,const TMatrixDSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv(TMatrixD &,const TMatrixDSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Double_t *sp  = source.GetMatrixArray();
        Double_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Double_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp  += i;
    trp += i;        // point to [i,i]
    tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      Assert(*sp != 0.0);
      if (j > i) *tcp /= *sp;
      *trp++ /= *sp++;
      tcp += ncols;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
void TMatrixD::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixD.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
    if (R__v > 2) {
      Clear();
      TMatrixD::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
    } else if (R__v == 2) { //process old version 2
      Clear();
      TObject::Streamer(R__b);
      MakeValid();
      R__b >> fNrows;
      R__b >> fNcols;
      R__b >> fNelems;
      R__b >> fRowLwb;
      R__b >> fColLwb;
      Char_t isArray;
      R__b >> isArray;
      if (isArray) {
        if (fNelems > 0) {
          fElements = new Double_t[fNelems];
          R__b.ReadFastArray(fElements,fNelems);
        } else
          fElements = 0;
      }
      R__b.CheckByteCount(R__s,R__c,TMatrixD::IsA());
    } else { //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      MakeValid();
      R__b >> fNrows;
      R__b >> fNcols;
      R__b >> fRowLwb;
      R__b >> fColLwb;
      fNelems = R__b.ReadArray(fElements);
      R__b.CheckByteCount(R__s,R__c,TMatrixD::IsA());
    }
    // in version <=2 , the matrix was stored column-wise
    if (R__v <= 2) {
      for (Int_t i = 0; i < fNrows; i++) {
        const Int_t off_i = i*fNcols;
        for (Int_t j = i; j < fNcols; j++) {
          const Int_t off_j = j*fNrows;
          const Double_t tmp = fElements[off_i+j];
          fElements[off_i+j] = fElements[off_j+i];
          fElements[off_j+i] = tmp;
        }
      }
    }
    if (fNelems > 0 && fNelems <= kSizeMax) {
      memcpy(fDataStack,fElements,fNelems*sizeof(Double_t));
      delete [] fElements;
      fElements = fDataStack;
    } else if (fNelems < 0)
      Invalidate();
  } else {
    TMatrixD::Class()->WriteBuffer(R__b,this);
  }
}
