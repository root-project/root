// @(#)root/matrix:$Name:  $:$Id: TMatrixF.cxx,v 1.5 2004/01/27 08:12:26 brun Exp $
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
// TMatrixF                                                             //
//                                                                      //
// Implementation of a general matrix in the linear algebra package     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixF.h"
#include "TMatrixFCramerInv.h"
#include "TDecompLU.h"

ClassImp(TMatrixF)

//______________________________________________________________________________
TMatrixF::TMatrixF(Int_t no_rows,Int_t no_cols)
{
  Allocate(no_rows,no_cols,0,0,1);
}

//______________________________________________________________________________
TMatrixF::TMatrixF(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb,1);
}

//______________________________________________________________________________
TMatrixF::TMatrixF(Int_t no_rows,Int_t no_cols,const Float_t *elements,Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]

  Allocate(no_rows,no_cols);
  SetMatrixArray(elements,option);
}

//______________________________________________________________________________
TMatrixF::TMatrixF(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                   const Float_t *elements,Option_t *option)
{
  Allocate(row_upb-row_lwb+1,col_upb-col_lwb+1,row_lwb,col_lwb);
  SetMatrixArray(elements,option);
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixF &another) : TMatrixFBase(another)
{
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());  
  *this = another;  
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixD &another)
{
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());  
  *this = another;  
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixFSym &another)
{
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());  
  *this = another;  
}

//______________________________________________________________________________
TMatrixF::TMatrixF(EMatrixCreatorsOp1 op,const TMatrixF &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Example: TMatrixF a(10,12); ...; TMatrixF b(TMatrixFBase::kTransposed, a);
  // Supported operations are: kZero, kUnit, kTransposed,  and kInverted .

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
      this->Invert();
      break;
    }

    case kAtA:
      AtMultB(prototype,prototype);
      break;

    default:
      Error("TMatrixF(EMatrixCreatorOp1)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixF &a,EMatrixCreatorsOp2 op,const TMatrixF &b)
{
  // Create a matrix applying a specific operation to two prototypes.
  // Example: TMatrixF a(10,12), b(12,5); ...; TMatrixF c(a, TMatrixFBase::kMult, b);
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

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      this->Invert();
      *this *= b;
      break;
    }

    default:
      Error("TMatrixF(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixF &a,EMatrixCreatorsOp2 op,const TMatrixFSym &b)
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

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      this->Invert();
      *this *= b;
      break;
    }

    default:
      Error("TMatrixF(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixFSym &a,EMatrixCreatorsOp2 op,const TMatrixF &b)
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

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      this->Invert();
      *this *= b;
      break;
    }

    default:
      Error("TMatrixF(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixFSym &a,EMatrixCreatorsOp2 op,const TMatrixFSym &b)
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

    case kInvMult:
    {
      Allocate(a.GetNrows(),a.GetNcols(),
               a.GetRowLwb(),a.GetColLwb(),1);
      *this = a;
      this->Invert();
      *this *= b;
      break;
    }

    default:
      Error("TMatrixF(EMatrixCreatorOp2)", "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixF::TMatrixF(const TMatrixFLazy &lazy_constructor)
{
  Allocate(lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetColUpb()-lazy_constructor.GetColLwb()+1,
           lazy_constructor.GetRowLwb(),lazy_constructor.GetColLwb(),1);
  lazy_constructor.FillIn(*this);
}

//______________________________________________________________________________
void TMatrixF::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,Int_t init)
{
  // Allocate new matrix. Arguments are number of rows, columns, row
  // lowerbound (0 default) and column lowerbound (0 default).

  Invalidate();

  if (no_rows <= 0 || no_cols <= 0)
  {
    Error("Allocate","no_rows=%d no_cols=%d",no_rows,no_cols);
    return;
  }

  fNrows   = no_rows;
  fNcols   = no_cols;
  fRowLwb  = row_lwb;
  fColLwb  = col_lwb;
  fNelems  = fNrows*fNcols;
  fIsOwner = kTRUE;
  fTol     = TMath::Sqrt(DBL_EPSILON);

  fElements = New_m(fNelems);
  if (init)
    memset(fElements,0,fNelems*sizeof(Float_t));
}

//______________________________________________________________________________
void TMatrixF::AMultB(const TMatrixF &a,const TMatrixF &b,Int_t constr)
{
  // General matrix multiplication. Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());

  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Float_t *ap = a.GetMatrixArray();
  const Float_t *bp = b.GetMatrixArray();
        Float_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,fNrows,fNcols,a.GetNcols(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t na     = a.GetNoElements();
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsb = b.GetNcols();
  const Float_t * const ap = a.GetMatrixArray();
  const Float_t * const bp = b.GetMatrixArray();
        Float_t *       cp = this->GetMatrixArray();

  const Float_t *arp0 = ap;                     // Pointer to  A[i,0];
  while (arp0 < ap+na) {
    for (const Float_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of B, Start bcp = B[0,0]
      const Float_t *arp = arp0;                       // Pointer to the i-th row of A, reset to A[i,0]
      Float_t cij = 0;
      while (bcp < bp+nb) {                     // Scan the i-th row of A and
        cij += *arp++ * *bcp;                   // the j-th col of B
        bcp += ncolsb;
      }
      *cp++ = cij;
      bcp -= nb-1;                              // Set bcp to the (j+1)-th col
    }
    arp0 += a.GetNcols();                       // Set ap to the (i+1)-th row
  }

  Assert(cp == this->GetMatrixArray()+fNelems && arp0 == ap+na);
#endif
}

//______________________________________________________________________________
void TMatrixF::AMultB(const TMatrixFSym &a,const TMatrixF &b,Int_t constr)
{
  // Matrix multiplication, with A symmetric and B general.
  // Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);      

  const Float_t *ap1 = a.GetMatrixArray();
  const Float_t *bp1 = b.GetMatrixArray();
        Float_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dsymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,b.GetNcols(),0.0,cp1,fNcols);
#else
  const Float_t *ap2 = a.GetMatrixArray();
  const Float_t *bp2 = b.GetMatrixArray();
        Float_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Float_t b_ij = *bp1++;
      *cp1 += b_ij*(*ap1);       
      Float_t tmp = 0.0;
      ap2 = ap1+1;
      for (Int_t k = i+1; k < fNrows; k++) {
        const Int_t index_kj = k*fNcols+j;
        const Float_t a_ik = *ap2++;
        const Float_t b_kj = bp2[index_kj];
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
void TMatrixF::AMultB(const TMatrixF &a,const TMatrixFSym &b,Int_t constr)
{
  // Matrix multiplication, with A general and B symmetric.
  // Create a matrix C such that C = A * B.
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNcols() != b.GetNrows() || a.GetColLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

  const Float_t *ap1 = a.GetMatrixArray();
        Float_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  const Float_t *bp1 = b.GetMatrixArray();
  cblas_dsymm (CblasRowMajor,CblasRight,CblasUpper,fNrows,fNcols,1.0,
               bp1,b.GetNcols(),ap1,a.GetNcols(),0.0,cp1,fNcols);
#else
  const Float_t *ap2 = a.GetMatrixArray();
  const Float_t *bp2 = b.GetMatrixArray();
        Float_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    const Float_t *bp1 = b.GetMatrixArray();
    for (Int_t j = 0; j < fNcols; j++) {
      const Float_t a_ij = *ap1++;
      *cp1 += a_ij*(*bp1);
      Float_t tmp = 0.0;
      ap2 = ap1;
      bp2 = bp1+1;
      cp2 = cp1+1;
      for (Int_t k = j+1; k < fNcols; k++) {
        const Float_t a_ik = *ap2++;
        const Float_t b_jk = *bp2++;
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
void TMatrixF::AMultB(const TMatrixFSym &a,const TMatrixFSym &b,Int_t constr)
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
  }

  if (constr)
    Allocate(a.GetNrows(),b.GetNcols(),a.GetRowLwb(),b.GetColLwb(),1);

  const Float_t *ap1 = a.GetMatrixArray();
  const Float_t *bp1 = b.GetMatrixArray();
        Float_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dsymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,b.GetNcols(),0.0,cp1,fNcols);
#else
  const Float_t *ap2 = a.GetMatrixArray();
  const Float_t *bp2 = b.GetMatrixArray();
        Float_t *cp2 = this->GetMatrixArray();
  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Float_t b_ij = *bp1++;
      *cp1 += b_ij*(*ap1);
      Float_t tmp = 0.0;
      ap2 = ap1+1;
      for (Int_t k = i+1; k < fNrows; k++) {
        const Int_t index_kj = k*fNcols+j;
        const Float_t a_ik = *ap2++;
        const Float_t b_kj = bp2[index_kj];
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
void TMatrixF::AtMultB(const TMatrixF &a,const TMatrixF &b,Int_t constr)
{
  // Create a matrix C such that C = A' * B. In other words,
  // c[i,j] = SUM{ a[k,i] * b[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNrows() != b.GetNrows() || a.GetRowLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
  }

  if (constr)
    Allocate(a.GetNcols(),b.GetNcols(),a.GetColLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Float_t *ap = a.GetMatrixArray();
  const Float_t *bp = b.GetMatrixArray();
        Float_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t nb     = b.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = b.GetNcols();
  const Float_t * const ap = a.GetMatrixArray();
  const Float_t * const bp = b.GetMatrixArray();
        Float_t *       cp = this->GetMatrixArray();

  const Float_t *acp0 = ap;           // Pointer to  A[i,0];
  while (acp0 < ap+a.GetNcols()) {
    for (const Float_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of B, Start bcp = B[0,0]
      const Float_t *acp = acp0;                       // Pointer to the i-th column of A, reset to A[0,i]
      Float_t cij = 0;
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
void TMatrixF::AtMultB(const TMatrixF &a,const TMatrixFSym &b,Int_t constr)
{
  // Create a matrix C such that C = A' * B. In other words,
  // c[i,j] = SUM{ a[k,i] * b[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());
  Assert(b.IsValid());
  if (a.GetNrows() != b.GetNrows() || a.GetRowLwb() != b.GetRowLwb()) {
    Error("AMultB","A rows and B columns incompatible");
    Invalidate();
  }

  if (constr)
    Allocate(a.GetNcols(),b.GetNcols(),a.GetColLwb(),b.GetColLwb(),1);

#ifdef CBLAS
  const Float_t *ap = a.GetMatrixArray();
  const Float_t *bp = b.GetMatrixArray();
        Float_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),bp,b.GetNcols(),1.0,cp,fNcols);
#else
  const Float_t *ap2 = a.GetMatrixArray();
  const Float_t *bp2 = b.GetMatrixArray();
        Float_t *cp1 = this->GetMatrixArray();
        Float_t *cp2 = this->GetMatrixArray();

  for (Int_t i = 0; i < fNrows; i++) {
    const Float_t *ap1 = a.GetMatrixArray()+i; // i-column of a
    const Float_t *bp1 = b.GetMatrixArray();
    for (Int_t j = 0; j < fNcols; j++) {
      const Float_t a_ji = *ap1;
      *cp1++ += a_ji*(*bp1);
      Float_t tmp = 0.0;
      ap2 = ap1;
      bp2 = bp1+1;
      cp2 = cp1;
      for (Int_t k = j+1; k < fNcols; k++) {
        const Float_t b_jk = *bp2++;
        *cp2 += a_ji*b_jk;
        tmp += (*ap1) * b_jk;
        ap1 += fNrows;
      }
      *cp1++ += tmp;
      ap1 += fNrows;
      bp1 += fNcols+1;
    }
  }
#endif
}

//______________________________________________________________________________
void TMatrixF::Adopt(Int_t nrows,Int_t ncols,Float_t *data)
{
  if (nrows <= 0 || nrows <= 0)
  {
    Error("Adopt","nrows=%d ncols=%d",nrows,ncols);
    return;
  }

  Clear();
  fNrows    = nrows;
  fNcols    = ncols;
  fRowLwb   = 0;
  fColLwb   = 0;
  fNelems   = fNrows*fNcols;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
void TMatrixF::Adopt(Int_t row_lwb,Int_t row_upb,
                     Int_t col_lwb,Int_t col_upb,Float_t *data)
{
  if (row_upb < row_lwb)
  {
    Error("Adopt","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
    return;
  }
  if (col_upb < col_lwb)
  {
    Error("Adopt","col_upb=%d < col_lwb=%d",col_upb,col_lwb);
    return;
  }

  Clear();
  fNrows    = row_upb-row_lwb+1;
  fNcols    = col_upb-col_lwb+1;
  fRowLwb   = row_lwb;
  fColLwb   = col_lwb;
  fNelems   = fNrows*fNcols;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
TMatrixF TMatrixF::GetSub(Int_t row_lwb,Int_t row_upb,
                          Int_t col_lwb,Int_t col_upb,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][col_lwb..col_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..col_upb-col_lwb+1] (default)
  // else          : return [row_lwb..row_upb][col_lwb..col_upb]

  Assert(IsValid());
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    return TMatrixFSym();
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("GetSub","col_lwb out of bounds");
    return TMatrixFSym();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    return TMatrixFSym();
  }
  if (col_upb < fColLwb || col_upb > fColLwb+fNcols-1) {
    Error("GetSub","col_upb out of bounds");
    return TMatrixFSym();
  }
  if (row_upb < row_lwb || col_upb < col_lwb) {
    Error("GetSub","row_upb < row_lwb || col_upb < col_lwb");
    return TMatrixFSym();
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t shift = (opt.Contains("S")) ? 1 : 0;

  Int_t row_lwb_sub;
  Int_t row_upb_sub;
  Int_t col_lwb_sub;
  Int_t col_upb_sub;
  if (shift) {
    row_lwb_sub = 0;
    row_upb_sub = row_upb-row_lwb;
    col_lwb_sub = 0;
    col_upb_sub = col_upb-col_lwb;
  } else {
    row_lwb_sub = row_lwb;
    row_upb_sub = row_upb;
    col_lwb_sub = col_lwb;
    col_upb_sub = col_upb;
  }

  TMatrixF sub(row_lwb_sub,row_upb_sub,col_lwb_sub,col_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;
  const Int_t ncols_sub = col_upb_sub-col_lwb_sub+1;

  const Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNcols+(col_lwb-fColLwb);
        Float_t *bp = sub.GetMatrixArray();

  for (Int_t irow = 0; irow < nrows_sub; irow++) {
    const Float_t *ap_sub = ap;
    for (Int_t icol = 0; icol < ncols_sub; icol++) {
      *bp++ = *ap_sub++;
    }
    ap += fNcols;
  }

  return sub;
}

//______________________________________________________________________________
void TMatrixF::SetSub(Int_t row_lwb,Int_t col_lwb,const TMatrixFBase &source)
{
  // Insert matrix source starting at [row_lwb][col_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source][col_lwb..col_lwb+ncols_source];

  Assert(IsValid());
  Assert(source.IsValid());

  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("SetSub","row_lwb outof bounds");
    return;
  }
  if (col_lwb < fColLwb || col_lwb > fColLwb+fNcols-1) {
    Error("SetSub","col_lwb outof bounds");
    return;
  }
  const Int_t nRows_source = source.GetNrows();
  const Int_t nCols_source = source.GetNcols();
  if (row_lwb+nRows_source > fRowLwb+fNrows || col_lwb+nCols_source > fColLwb+fNcols) {
    Error("SetSub","source matrix too large");
    return;
  }

  const Float_t *bp = source.GetMatrixArray();
        Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNcols+(col_lwb-fColLwb);

  for (Int_t irow = 0; irow < nRows_source; irow++) {
    Float_t *ap_sub = ap;
    for (Int_t icol = 0; icol < nCols_source; icol++) {
      *ap_sub++ = *bp++;
    }
    ap += fNcols;
  }
}

//______________________________________________________________________________
Double_t TMatrixF::Determinant() const
{
  const TMatrixD tmp(*this);
  TDecompLU lu(tmp,(Double_t)fTol);
  Double_t d1,d2;
  lu.Det(d1,d2);
  return d1*TMath::Power(2.0,d2);
}

//______________________________________________________________________________
void TMatrixF::Determinant(Double_t &d1,Double_t &d2) const
{
  const TMatrixD tmp(*this);
  TDecompLU lu(tmp,(Double_t)fTol);
  lu.Det(d1,d2);
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Zero()
{
  Assert(IsValid());
  memset(this->GetMatrixArray(),0,fNelems*sizeof(Float_t));

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Abs()
{
  // Take an absolute value of a matrix, i.e. apply Abs() to each element.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNelems;
  while (ep < fp) {
    *ep = TMath::Abs(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Sqr()
{
  // Square each element of the matrix.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNelems;
  while (ep < fp) {
    *ep = (*ep) * (*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Sqrt()
{
  // Take square root of all elements.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNelems;
  while (ep < fp) {
    *ep = TMath::Sqrt(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::UnitMatrix()
{
  // Make a unit matrix (matrix need not be a square one).

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  memset(ep,0,fNelems*sizeof(Float_t));
  for (Int_t i = fRowLwb; i <= fRowLwb+fNrows-1; i++)
    for (Int_t j = fColLwb; j <= fColLwb+fNcols-1; j++)
      *ep++ = (i==j ? 1.0 : 0.0);

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Invert(Double_t *det)
{
  // Invert the matrix and calculate its determinant

  Assert(IsValid());

  if (GetNrows() != GetNcols() || GetRowLwb() != GetColLwb()) {
    Error("Invert()","matrix should be square");
    Invalidate();
    return *this;
  }

  Int_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Int_t *index = work;
  if (fNcols > kWorkMax) {
    isAllocated = kTRUE;
    index = new Int_t[fNcols];
  }

  TMatrixD tmp(*this);
  Double_t sign = 1.0;
  Int_t nrZeros = 0;
  TDecompLU::DecomposeLU(tmp,index,sign,(Double_t)fTol,nrZeros);
  if (det) {
    Double_t d1;
    Double_t d2;
    const TVectorD diagv = TMatrixDDiag_const(tmp);
    TDecompBase::DiagProd(diagv,(Double_t)fTol,d1,d2);
    d1 *= sign;
    if (TMath::Abs(d2) > 52.0) {
      Error("Invert(Double_t *)","Determinant under/over-flows double: det= %.4f 2^%.0f",d1,d2);
      *det =  0.0;
    } else
      *det = d1*TMath::Power(2.0,d2);
  }

  TDecompLU::InvertLU(tmp,index,fTol);
  *this = tmp;

  if (isAllocated)
    delete [] index;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::InvertFast(Double_t *det)
{
  // Invert the matrix and calculate its determinant

  Assert(IsValid());

  if (GetNrows() != GetNcols() || GetRowLwb() != GetColLwb()) {
    Error("Invert()","matrix should be square");
    Invalidate();
    return *this;
  }

  const Char_t nRows = Char_t(GetNrows());
  switch (nRows) {
    case 1:
    {
      Float_t *pM = this->GetMatrixArray();
      if (*pM == 0.) Invalidate();
      else           *pM = 1.0/(*pM);
      return *this;
    }
    case 2:
    {
      if (!TMatrixFCramerInv::Inv2x2(*this,det))
        Invalidate();
      return *this;
    }
    case 3:
    {
      if (!TMatrixFCramerInv::Inv3x3(*this,det))
        Invalidate();
      return *this;
    }
    case 4:
    {
      if (!TMatrixFCramerInv::Inv4x4(*this,det))
        Invalidate();
      return *this;
    }
    case 5:
    {
      if (!TMatrixFCramerInv::Inv5x5(*this,det))
        Invalidate();
      return *this;
    }
    case 6:
    {
      if (!TMatrixFCramerInv::Inv6x6(*this,det))
        Invalidate();
      return *this;
    }

    default:
    {
      Int_t work[kWorkMax];
      Bool_t isAllocated = kFALSE;
      Int_t *index = work;
      if (fNcols > kWorkMax) {
        isAllocated = kTRUE;
        index = new Int_t[fNcols];
      }

      TMatrixD tmp(*this);
      Double_t sign = 1.0;
      Int_t nrZeros;
      TDecompLU::DecomposeLU(tmp,index,sign,(Double_t)fTol,nrZeros);
      if (det) {
        Double_t d1;
        Double_t d2;
        const TVectorD diagv = TMatrixDDiag_const(tmp);
        TDecompBase::DiagProd(diagv,(Double_t)fTol,d1,d2);
        d1 *= sign;
        if (TMath::Abs(d2) > 52.0) {
          Error("Invert(Double_t *)","Determinant under/over-flows double: det= %.4f 2^%.0f",d1,d2);
          *det =  0.0;
        } else
          *det = d1*TMath::Power(2.0,d2);
      }

      TDecompLU::InvertLU(tmp,index,(Double_t)fTol);
      *this = tmp;

      if (isAllocated)
        delete [] index;

      return *this;
    }
  }
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Transpose(const TMatrixF &source)
{
  // Transpose a matrix.

  Assert(IsValid());
  Assert(source.IsValid());

  if (this == &source) {
    Float_t *ap = this->GetMatrixArray();
    if (fNrows == fNcols && fRowLwb == 0 && fColLwb == 0) {
      for (Int_t i = 0; i < fNrows; i++) {
        const Int_t off_i = i*fNrows;
        for (Int_t j = i+1; j < fNcols; j++) {
          const Int_t off_j = j*fNcols;
          const Float_t tmp = ap[off_i+j];
          ap[off_i+j] = ap[off_j+i];
          ap[off_j+i] = tmp;
        }
      }
    } else {
      const TMatrixF oldMat = source;
      Int_t tmp;
      tmp = fNrows;  fNrows  = fNcols;  fNcols  = tmp;
      tmp = fRowLwb; fRowLwb = fColLwb; fColLwb = tmp;
      for (Int_t irow = fRowLwb; irow < fRowLwb+fNrows; irow++) {
        for (Int_t icol = fColLwb; icol < fColLwb+fNcols; icol++) {
          (*this)(irow,icol) = oldMat(icol,irow);
        }
      }
    }
  } else {
    if (fNrows  != source.GetNcols()  || fNcols  != source.GetNrows() ||
        fRowLwb != source.GetColLwb() || fColLwb != source.GetRowLwb())
    {
      Error("Transpose","matrix has wrong shape");
      Invalidate();
      return *this;
    }

    const Float_t *sp1 = source.GetMatrixArray();
    const Float_t *scp = sp1; // Row source pointer
          Float_t *tp  = this->GetMatrixArray();
    const Float_t * const tp_last = this->GetMatrixArray()+fNelems;

    // (This: target) matrix is traversed row-wise way,
    // whilst the source matrix is scanned column-wise
    while (tp < tp_last) {
      const Float_t *sp2 = scp++;

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
TMatrixF &TMatrixF::NormByDiag(const TVectorF &v,Option_t *option)
{
  // option:
  // "D"   :  b(i,j) = a(i,j)/sqrt(abs*(v(i)*v(j)))  (default)
  // else  :  b(i,j) = a(i,j)*sqrt(abs*(v(i)*v(j)))  (default)

  Assert(IsValid());
  Assert(v.IsValid());

  const Int_t nMax = TMath::Max(fNrows,fNcols);
  if (v.GetNoElements() < nMax) {
    Error("NormByDiag","vector shorter than matrix diagonal");
    Invalidate();
    return *this;
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t divide = (opt.Contains("D")) ? 1 : 0;

  const Float_t* pV = v.GetMatrixArray();
        Float_t *mp = this->GetMatrixArray();

  if (divide) {
    for (Int_t irow = 0; irow < fNrows; irow++) {
      for (Int_t icol = 0; icol < fNcols; icol++) {
        const Float_t val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
        Assert(val != 0.0);
        *mp++ /= val;
      }
    }
  } else {
    for (Int_t irow = 0; irow < fNrows; irow++) {
      for (Int_t icol = 0; icol < fNcols; icol++) {
        const Float_t val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
        *mp++ *= val;
      }
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::NormByColumn(const TVectorF &v,Option_t *option)
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

  const Float_t* pv = v.GetMatrixArray();
        Float_t *mp = this->GetMatrixArray();
  const Float_t * const mp_last = mp+fNelems;

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
TMatrixF &TMatrixF::NormByRow(const TVectorF &v,Option_t *option)
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

  const Float_t *pv0 = v.GetMatrixArray();
  const Float_t *pv  = pv0;
        Float_t *mp  = this->GetMatrixArray();
  const Float_t * const mp_last = mp+fNelems;

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
TMatrixF &TMatrixF::operator=(const TMatrixF &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixF &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNelems*sizeof(Float_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator=(const TMatrixD &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixF &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (dynamic_cast<TMatrixD *>(this) != &source) {
    TObject::operator=(source);
    const Double_t * const ps = source.GetMatrixArray();
          Float_t  * const pt = GetMatrixArray();
    for (Int_t i = 0; i < fNelems; i++)
      pt[i] = (Float_t) ps[i];
    fTol = (Float_t)source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator=(const TMatrixFSym &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TMatrixFSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  if ((TMatrixFBase *)this != (TMatrixFBase *)&source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNelems*sizeof(Float_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator=(const TMatrixFLazy &lazy_constructor)
{
  Assert(IsValid());

  if (lazy_constructor.GetRowUpb() != GetRowUpb() ||
      lazy_constructor.GetColUpb() != GetColUpb() ||
      lazy_constructor.GetRowLwb() != GetRowLwb() ||
      lazy_constructor.GetColLwb() != GetColLwb()) {
    Error("operator=(const TMatrixFLazy&)", "matrix is incompatible with "
          "the assigned Lazy matrix");
    Invalidate();
    return *this;
  }

  lazy_constructor.FillIn(*this);
  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator=(Float_t val)
{
  // Assign val to every element of the matrix.

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator+=(Float_t val)
{
  // Add val to every element of the matrix.

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator-=(Float_t val)
{
  // Subtract val from every element of the matrix.

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator*=(Float_t val)
{
  // Multiply every element of the matrix with val.

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator+=(const TMatrixF &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TMatrixF &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp = source.GetMatrixArray();
  Float_t *tp = this->GetMatrixArray();
  const Float_t * const tp_last = tp+fNelems;
  while (tp < tp_last)
    *tp++ += *sp++;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator+=(const TMatrixFSym &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TMatrixFSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp  = source.GetMatrixArray();
        Float_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Float_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
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
TMatrixF &TMatrixF::operator-=(const TMatrixF &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator=-(const TMatrixF &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp = source.GetMatrixArray();
  Float_t *tp = this->GetMatrixArray();
  const Float_t * const tp_last = tp+fNelems;
  while (tp < tp_last)
    *tp++ -= *sp++;

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator-=(const TMatrixFSym &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator=-(const TMatrixFSym &)","matrices not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp = source.GetMatrixArray();
        Float_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Float_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
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
TMatrixF &TMatrixF::operator*=(const TMatrixF &source)
{
  // Compute target = target * source inplace. Strictly speaking, it can't be
  // done inplace, though only the row of the target matrix needs to be saved.
  // "Inplace" multiplication is only allowed when the 'source' matrix is square.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNcols != source.GetNrows() || fColLwb != source.GetRowLwb() ||
      fNcols != source.GetNcols() || fColLwb != source.GetColLwb()) {
    Error("operator*=(const TMatrixF &)","source matrix has wrong shape");
    Invalidate();
    return *this;
  }

  // Check for A *= A;
  const Float_t *sp;
  TMatrixF tmp;
  if (this == &source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  // One row of the old_target matrix
  Float_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Float_t *trp = work;
  if (fNcols > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Float_t[fNcols];
  }

        Float_t *cp   = this->GetMatrixArray();
  const Float_t *trp0 = cp; // Pointer to  target[i,0];
  const Float_t * const trp0_last = trp0+fNelems;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcols*sizeof(Float_t));        // copy the i-th row of target, Start at target[i,0]
    for (const Float_t *scp = sp; scp < sp+fNcols; ) {  // Pointer to the j-th column of source,
                                                         // Start scp = source[0,0]
      Float_t cij = 0;
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
TMatrixF &TMatrixF::operator*=(const TMatrixFSym &source)
{
  // Compute target = target * source inplace. Strictly speaking, it can't be
  // done inplace, though only the row of the target matrix needs to be saved.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNcols != source.GetNrows() || fColLwb != source.GetRowLwb()) {
    Error("operator*=(const TMatrixFSym &)","source matrix has wrong shape");
    Invalidate();
    return *this;
  }

  // Check for A *= A;
  const Float_t *sp;
  TMatrixF tmp;
  if ((TMatrixFBase *)this == (TMatrixFBase *)&source) {
    tmp.ResizeTo(source);
    tmp = source;
    sp = tmp.GetMatrixArray();
  }
  else
    sp = source.GetMatrixArray();

  // One row of the old_target matrix
  Float_t work[kWorkMax];
  Bool_t isAllocated = kFALSE;
  Float_t *trp = work;
  if (fNcols > kWorkMax) {
    isAllocated = kTRUE;
    trp = new Float_t[fNcols];
  }

        Float_t *cp   = this->GetMatrixArray();
  const Float_t *trp0 = cp; // Pointer to  target[i,0];
  const Float_t * const trp0_last = trp0+fNelems;
  while (trp0 < trp0_last) {
    memcpy(trp,trp0,fNcols*sizeof(Float_t));        // copy the i-th row of target, Start at target[i,0]
    for (const Float_t *scp = sp; scp < sp+fNcols; ) {   // Pointer to the j-th column of source, 
                                                                     //Start scp = source[0,0]
      Float_t cij = 0;
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
TMatrixF &TMatrixF::operator*=(const TMatrixFDiag_const &diag)
{
  // Multiply a matrix row by the diagonal of another matrix
  // matrix(i,j) *= diag(j), j=1,fNcols

  Assert(IsValid());
  Assert(diag.GetMatrix()->IsValid());
  Assert(fNcols == diag.GetNdiags());

  if (fNcols != diag.GetNdiags()) {
    Error("operator*=(const TMatrixFDiag_const &)","wrong diagonal length");
    Invalidate();
    return *this;
  }

  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Int_t inc = diag.GetInc();
  while (mp < mp_last) {
    const Float_t *dp = diag.GetPtr();
    for (Int_t j = 0; j < fNcols; j++) {
      *mp++ *= *dp;
      dp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator/=(const TMatrixFDiag_const &diag)
{
  // Divide a matrix row by the diagonal of another matrix
  // matrix(i,j) /= diag(j)

  Assert(IsValid());
  Assert(diag.GetMatrix()->IsValid());

  if (fNcols != diag.GetNdiags()) {
    Error("operator/=(const TMatrixFDiag_const &)","wrong diagonal length");
    Invalidate();
    return *this;
  }

  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Int_t inc = diag.GetInc();
  while (mp < mp_last) {
    const Float_t *dp = diag.GetPtr();
    for (Int_t j = 0; j < fNcols; j++) {
      Assert(*dp != 0.0);
      *mp++ /= *dp;
      dp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator*=(const TMatrixFColumn_const &col)
{
  // Multiply a matrix by the column of another matrix
  // matrix(i,j) *= another(i,k) for fixed k

  const TMatrixFBase *mt = col.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNrows != mt->GetNrows()) {
    Error("operator*=(const TMatrixFColumn_const &)","wrong column length");
    Invalidate();
    return *this;
  }

  const Float_t * const endp = col.GetPtr()+mt->GetNoElements();
  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Float_t *cp = col.GetPtr();      //  ptr
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
TMatrixF &TMatrixF::operator/=(const TMatrixFColumn_const &col)
{
  // Divide a matrix by the column of another matrix
  // matrix(i,j) /= another(i,k) for fixed k

  const TMatrixFBase *mt = col.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNrows != mt->GetNrows()) {
    Error("operator/=(const TMatrixFColumn_const &)","wrong column matrix");
    Invalidate();
    return *this;
  }

  const Float_t * const endp = col.GetPtr()+mt->GetNoElements();
  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Float_t *cp = col.GetPtr();      //  ptr
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
TMatrixF &TMatrixF::operator*=(const TMatrixFRow_const &row)
{
  // Multiply a matrix by the row of another matrix
  // matrix(i,j) *= another(k,j) for fixed k

  const TMatrixFBase *mt = row.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNcols != mt->GetNcols()) {
    Error("operator*=(const TMatrixFRow_const &)","wrong row length");
    Invalidate();
    return *this;
  }

  const Float_t * const endp = row.GetPtr()+mt->GetNoElements();
  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Int_t inc = row.GetInc();
  while (mp < mp_last) {
    const Float_t *rp = row.GetPtr();    // Row ptr
    for (Int_t j = 0; j < fNcols; j++) {
      Assert(rp < endp);
      *mp++ *= *rp;
      rp += inc;
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::operator/=(const TMatrixFRow_const &row)
{
  // Divide a matrix by the row of another matrix
  // matrix(i,j) /= another(k,j) for fixed k

  const TMatrixFBase *mt = row.GetMatrix();
  Assert(IsValid());
  Assert(mt->IsValid());

  if (fNcols != mt->GetNcols()) {
    Error("operator/=(const TMatrixFRow_const &)","wrong row length");
    Invalidate();
    return *this;
  }

  const Float_t * const endp = row.GetPtr()+mt->GetNoElements();
  Float_t *mp = this->GetMatrixArray();  // Matrix ptr
  const Float_t * const mp_last = mp+fNelems;
  const Int_t inc = row.GetInc();
  while (mp < mp_last) {
    const Float_t *rp = row.GetPtr();    // Row ptr
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
TMatrixF &TMatrixF::Apply(const TElementActionF &action)
{
  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    action.Operation(*ep++);

  return *this;
}

//______________________________________________________________________________
TMatrixF &TMatrixF::Apply(const TElementPosActionF &action)
{
  // Apply action to each element of the matrix. To action the location
  // of the current element is passed.

  Assert(IsValid());

  Float_t *ep = this->GetMatrixArray();
  for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
    for (action.fJ = fColLwb; action.fJ < fColLwb+fNcols; action.fJ++)
      action.Operation(*ep++);

  Assert(ep == this->GetMatrixArray()+fNelems);

  return *this;
}

//______________________________________________________________________________
Bool_t operator==(const TMatrixF &m1,const TMatrixF &m2)
{
  // Check to see if two matrices are identical.

  if (!AreCompatible(m1,m2)) return kFALSE;
  return (memcmp(m1.GetMatrixArray(),m2.GetMatrixArray(),
                 m1.GetNoElements()*sizeof(Float_t)) == 0);
}

//______________________________________________________________________________
TMatrixF operator+(const TMatrixF &source1,const TMatrixF &source2)
{
  TMatrixF target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixF operator+(const TMatrixF &source1,const TMatrixFSym &source2)
{
  TMatrixF target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixF operator+(const TMatrixFSym &source1,const TMatrixF &source2)
{
  TMatrixF target(source2);
  target += source1;
  return target;
}

//______________________________________________________________________________
TMatrixF operator-(const TMatrixF &source1,const TMatrixF &source2)
{
  TMatrixF target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixF operator-(const TMatrixF &source1,const TMatrixFSym &source2)
{
  TMatrixF target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixF operator-(const TMatrixFSym &source1,const TMatrixF &source2)
{
  TMatrixF target(source2);
  target -= source1;
  return target;
}

//______________________________________________________________________________
TMatrixF operator*(Float_t val,const TMatrixF &source)
{
  TMatrixF target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixF operator*(const TMatrixF &source1,const TMatrixF &source2)
{
  TMatrixF target(source1,TMatrixF::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixF operator*(const TMatrixF &source1,const TMatrixFSym &source2)
{
  TMatrixF target(source1,TMatrixF::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixF operator*(const TMatrixFSym &source1,const TMatrixF &source2)
{
  TMatrixF target(source1,TMatrixF::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixF operator*(const TMatrixFSym &source1,const TMatrixFSym &source2)
{
  TMatrixF target(source1,TMatrixF::kMult,source2);
  return target;
}

//______________________________________________________________________________
TMatrixF &Add(TMatrixF &target,Float_t scalar,const TMatrixF &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    ::Error("Add(TMatrixF &,Float_t,const TMatrixF &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Float_t *sp  = source.GetMatrixArray();
        Float_t *tp  = target.GetMatrixArray();
  const Float_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp )
    *tp++ += scalar * (*sp++);

  return target;
}

//______________________________________________________________________________
TMatrixF &Add(TMatrixF &target,Float_t scalar,const TMatrixFSym &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    ::Error("Add(TMatrixF &,Float_t,const TMatrixFSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Float_t *sp  = source.GetMatrixArray();
        Float_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Float_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < nrows; i++) {
    sp  += i;
    trp += i;        // point to [i,i]
    tcp += i*ncols;  // point to [i,i]
    for (Int_t j = i; j < ncols; j++) {
      const Float_t tmp = scalar * *sp++;
      if (j > i) *tcp += tmp;
      tcp += ncols;
      *trp++ += tmp;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixF &ElementMult(TMatrixF &target,const TMatrixF &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult(TMatrixF &,const TMatrixF &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Float_t *sp  = source.GetMatrixArray();
        Float_t *tp  = target.GetMatrixArray();
  const Float_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp )
    *tp++ *= *sp++;

  return target;
}

//______________________________________________________________________________
TMatrixF &ElementMult(TMatrixF &target,const TMatrixFSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult(TMatrixF &,const TMatrixFSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Float_t *sp  = source.GetMatrixArray();
        Float_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Float_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
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
TMatrixF &ElementDiv(TMatrixF &target,const TMatrixF &source)
{
  // Divide target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv(TMatrixF &,const TMatrixF &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Float_t *sp  = source.GetMatrixArray();
        Float_t *tp  = target.GetMatrixArray();
  const Float_t *ftp = tp+target.GetNoElements();
  while ( tp < ftp ) {
    Assert(*sp != 0.0);
    *tp++ /= *sp++;
  }

  return target;
}

//______________________________________________________________________________
TMatrixF &ElementDiv(TMatrixF &target,const TMatrixFSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv(TMatrixF &,const TMatrixFSym &)","matrices not compatible");
    target.Invalidate();
    return target;
  }

  const Int_t nrows   = target.GetNrows();
  const Int_t ncols   = target.GetNcols();
  const Int_t nelems  = target.GetNoElements();
  const Float_t *sp  = source.GetMatrixArray();
        Float_t *trp = target.GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
        Float_t *tcp = target.GetMatrixArray(); // pointer to LL part,              traverse col-wise
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
void TMatrixF::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixF.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
    if (R__v > 1) {
      Clear();
      TMatrixF::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      if (fNelems <= kSizeMax) {
        memcpy(fDataStack,fElements,fNelems*sizeof(Float_t));
        delete [] fElements;
        fElements = fDataStack;
      }
      return;
    }
    //====process old versions before automatic schema evolution
    TObject::Streamer(R__b);
    fNelems = R__b.ReadArray(fElements);
    R__b.CheckByteCount(R__s,R__c,TMatrixF::IsA());
    //====end of old versions
  } else {
    TMatrixF::Class()->WriteBuffer(R__b,this);
  }
}
