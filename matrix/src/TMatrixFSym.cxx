// @(#)root/matrix:$Name:  $:$Id: TMatrixFSym.cxx,v 1.6 2004/04/15 09:21:51 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFSym                                                          //
//                                                                      //
// Implementation of a symmetric matrix in the linear algebra package   //
//                                                                      //
// Note that in this implementation both matrix element m[i][j] and     //
// m[j][i] are updated and stored in memory . However, when making the  //
// object persistent only the upper right triangle is stored .          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixFSym.h"
#include "TDecompLU.h"
#include "TMatrixDSymEigen.h"

ClassImp(TMatrixFSym)

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(Int_t no_rows)
{
  Allocate(no_rows,no_rows,0,0,1);
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(Int_t row_lwb,Int_t row_upb)
{
  const Int_t no_rows = row_upb-row_lwb+1;
  Allocate(no_rows,no_rows,row_lwb,row_lwb,1);
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(Int_t no_rows,const Float_t *elements,Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]

  Allocate(no_rows,no_rows);
  SetMatrixArray(elements,option);
  if (!IsSymmetric()) {
    Error("TMatrixFSym(Int_t,Float_t*,Option_t*)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(Int_t row_lwb,Int_t row_upb,const Float_t *elements,Option_t *option)
{
  const Int_t no_rows = row_upb-row_lwb+1;
  Allocate(no_rows,no_rows,row_lwb,row_lwb);
  SetMatrixArray(elements,option);
  if (!IsSymmetric()) {
    Error("TMatrixFSym(Int_t,Int_t,Float_t*,Option_t*)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(const TMatrixFSym &another) : TMatrixFBase(another)
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(EMatrixCreatorsOp1 op,const TMatrixFSym &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Example: TMatrixFSym a(10,12); ...; TMatrixFSym b(TMatrixFBase::kTransposed, a);
  // Supported operations are: kZero, kUnit, and kTransposed .

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

    case kAtA:
      AtMultA(prototype);
      break;

    default:
      Error("TMatrixFSym(EMatrixCreatorOp1,const TMatrixFSym)",
             "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(EMatrixCreatorsOp1 op,const TMatrixF &prototype)
{
  Assert(dynamic_cast<TMatrixF *>(this) != &prototype);
  Invalidate();
                   
  Assert(prototype.IsValid());

  switch(op) {
    case kAtA:
      AtMultA(prototype);
      break;

    default:
      Error("TMatrixFSym(EMatrixCreatorOp1,const TMatrixF)",
             "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixFSym::TMatrixFSym(const TMatrixFSymLazy &lazy_constructor)
{
  Allocate(lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetRowLwb(),lazy_constructor.GetRowLwb(),1);
  lazy_constructor.FillIn(*this);
  if (!IsSymmetric()) {
    Error("TMatrixFSym(TMatrixFSymLazy)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
void TMatrixFSym::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,
                           Int_t init,Int_t /*nr_nonzeros*/)
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
  fTol     = DBL_EPSILON;

  fElements = New_m(fNelems);
  if (init)
    memset(fElements,0,fNelems*sizeof(Float_t));
}

//______________________________________________________________________________
void TMatrixFSym::AtMultA(const TMatrixF &a,Int_t constr)
{
  // Create a matrix C such that C = A' * A. In other words,
  // c[i,j] = SUM{ a[k,i] * a[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());

  if (constr)
    Allocate(a.GetNcols(),a.GetNcols(),a.GetColLwb(),a.GetColLwb(),1);

#ifdef CBLAS
  const Float_t *ap = a.GetMatrixArray();
        Float_t *cp = this->GetMatrixArray();
  cblas_sgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),ap,a.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t nb     = a.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = ncolsa;
  const Float_t * const ap = a.GetMatrixArray();
  const Float_t * const bp = ap;
        Float_t *       cp = this->GetMatrixArray();

  const Float_t *acp0 = ap;           // Pointer to  A[i,0];
  while (acp0 < ap+a.GetNcols()) {
    for (const Float_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of A, Start bcp = A[0,0]
      const Float_t *acp = acp0;                       // Pointer to the i-th column of A, reset to A[0,i]
      Float_t cij = 0;
      while (bcp < bp+nb) {           // Scan the i-th column of A and
        cij += *acp * *bcp;           // the j-th col of A
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
void TMatrixFSym::AtMultA(const TMatrixFSym &a,Int_t constr)
{
  // Matrix multiplication, with A symmetric
  // Create a matrix C such that C = A' * A = A * A = A * A'
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());

  if (constr)
    Allocate(a.GetNcols(),a.GetNcols(),a.GetColLwb(),a.GetColLwb(),1);

  const Float_t *ap1 = a.GetMatrixArray();
  const Float_t *bp1 = ap1;
        Float_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_ssymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,a.GetNcols(),0.0,cp1,fNcols);
#else
  const Float_t *ap2 = a.GetMatrixArray();
  const Float_t *bp2 = ap2;
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
void TMatrixFSym::Use(Int_t nrows,Float_t *data)
{
  if (nrows <= 0)
  {
    Error("Use","nrows=%d",nrows);
    return;
  }
  
  Clear();
  fNrows    = nrows;
  fNcols    = nrows;
  fRowLwb   = 0;
  fColLwb   = 0;
  fNelems   = fNrows*fNcols;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________ 
void TMatrixFSym::Use(Int_t row_lwb,Int_t row_upb,Float_t *data)
{
  if (row_upb < row_lwb)
  {
    Error("Use","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
    return;
  }

  Clear();
  fNrows    = row_upb-row_lwb+1;
  fNcols    = fNrows;
  fRowLwb   = row_lwb;
  fColLwb   = row_lwb;
  fNelems   = fNrows*fNcols;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
TMatrixFSym TMatrixFSym::GetSub(Int_t row_lwb,Int_t row_upb,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][row_lwb..row_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..row_upb-row_lwb+1] (default)
  // else          : return [row_lwb..row_upb][row_lwb..row_upb]

  Assert(IsValid());

  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    return TMatrixFSym();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    return TMatrixFSym();
  }
  if (row_upb < row_lwb) {
    Error("GetSub","row_upb < row_lwb");
    return TMatrixFSym();
  }

  TString opt(option);
  opt.ToUpper();
  const Int_t shift = (opt.Contains("S")) ? 1 : 0;

  Int_t row_lwb_sub;
  Int_t row_upb_sub;
  if (shift) {
    row_lwb_sub = 0;
    row_upb_sub = row_upb-row_lwb;
  } else {
    row_lwb_sub = row_lwb;
    row_upb_sub = row_upb;
  }

  TMatrixFSym sub(row_lwb_sub,row_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;

  const Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNrows+(row_lwb-fRowLwb);
        Float_t *bp = sub.GetMatrixArray();

  for (Int_t irow = 0; irow < nrows_sub; irow++) {
    const Float_t *ap_sub = ap;
    for (Int_t icol = 0; icol < nrows_sub; icol++) {
      *bp++ = *ap_sub++;
    }
    ap += fNrows;
  }

  return sub;
}

//______________________________________________________________________________
void TMatrixFSym::SetSub(Int_t row_lwb,const TMatrixFSym &source)
{
  // Insert matrix source starting at [row_lwb][row_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source][row_lwb..row_lwb+nrows_source];

  Assert(IsValid());
  Assert(source.IsValid());

  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("SetSub","row_lwb outof bounds");
    return;
  }
  const Int_t nRows_source = source.GetNrows();
  if (row_lwb+nRows_source > fRowLwb+fNrows) {
    Error("SetSub","source matrix too large");
    return;
  }

  const Float_t *bp = source.GetMatrixArray();
        Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNrows+(row_lwb-fRowLwb);

  for (Int_t irow = 0; irow < nRows_source; irow++) {
    Float_t *ap_sub = ap;
    for (Int_t icol = 0; icol < nRows_source; icol++) {
      *ap_sub++ = *bp++;
    }
    ap += fNrows;
  }
}

//______________________________________________________________________________
void TMatrixFSym::SetMatrixArray(const Float_t *data,Option_t *option)
{
  TMatrixFBase::SetMatrixArray(data,option);
  if (!this->IsSymmetric()) {
    Error("SetMatrixArray","Matrix is not symmetric after Set");
    Invalidate();
  }
}

//______________________________________________________________________________
void TMatrixFSym::Shift(Int_t row_shift,Int_t col_shift)
{
  if (row_shift != col_shift) {
    Error("Shift","row_shift != col_shift");
    Invalidate();
  }
  TMatrixFBase::Shift(row_shift,col_shift);
}

//______________________________________________________________________________
void TMatrixFSym::ResizeTo(Int_t nrows,Int_t ncols,Int_t /*nr_nonzeros*/)
{
  if (nrows != ncols) {
    Error("ResizeTo","nrows != ncols");
    Invalidate();
  }
  TMatrixFBase::ResizeTo(nrows,ncols);
}

//______________________________________________________________________________
void TMatrixFSym::ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                           Int_t /*nr_nonzeros*/)
{
  if (row_lwb != col_lwb) {
    Error("ResizeTo","row_lwb != col_lwb");
    Invalidate();
  }
  if (row_upb != col_upb) {
    Error("ResizeTo","row_upb != col_upb");
    Invalidate();
  }
  TMatrixFBase::ResizeTo(row_lwb,row_upb,col_lwb,col_upb);
}

//______________________________________________________________________________
Double_t TMatrixFSym::Determinant() const
{
  const TMatrixD tmp = TMatrixDSym(*this);
  TDecompLU lu(tmp,fTol);
  Double_t d1,d2;
  lu.Det(d1,d2);
  return d1*TMath::Power(2.0,d2);
}

//______________________________________________________________________________
void TMatrixFSym::Determinant(Double_t &d1,Double_t &d2) const
{
  const TMatrixD tmp = TMatrixDSym(*this);
  TDecompLU lu(tmp,fTol);
  lu.Det(d1,d2);
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::Transpose(const TMatrixFSym &source)
{
  // Transpose a matrix.

  Assert(IsValid());
  Assert(source.IsValid());

  if (fNrows != source.GetNcols() || fRowLwb != source.GetColLwb())
  {
    Error("Transpose","matrix has wrong shape");
    Invalidate();
    return *this;
  }

  *this = source;
  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator=(const TMatrixFSym &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(this->GetMatrixArray(),source.fElements,fNelems*sizeof(Float_t));
    fTol = source.GetTol();
  }
  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator=(const TMatrixFSymLazy &lazy_constructor)
{
  Assert(IsValid());

  if (lazy_constructor.fRowUpb != GetRowUpb() ||
      lazy_constructor.fRowLwb != GetRowLwb()) {
     Error("operator=(const TMatrixFSymLazy&)", "matrix is incompatible with "
           "the assigned Lazy matrix");
    Invalidate();
    return *this;
  }

  lazy_constructor.FillIn(*this);
  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator=(Float_t val)
{
  // Assign val to every element of the matrix.

  Assert(IsValid());

  Float_t *ep = fElements;
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator+=(Float_t val)
{
  // Add val to every element of the matrix.

  Assert(IsValid());

  Float_t *ep = fElements;
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator-=(Float_t val)
{
  // Subtract val from every element of the matrix.

  Assert(IsValid());

  Float_t *ep = fElements;
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator*=(Float_t val)
{
  // Multiply every element of the matrix with val.

  Assert(IsValid());

  Float_t *ep = fElements;
  const Float_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::operator+=(const TMatrixFSym &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=","matrices not compatible");
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
TMatrixFSym &TMatrixFSym::operator-=(const TMatrixFSym &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator-=","matrices not compatible");
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
      if (j > i) *tcp -= *sp;
      *trp++ -= *sp++;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::Apply(const TElementActionF &action)
{ 
  Assert(IsValid());
  
  Float_t val = 0;
  Float_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
  Float_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < fNrows; i++) {
    trp += i;         // point to [i,i]
    tcp += i*fNcols;  // point to [i,i]
    for (Int_t j = i; j < fNcols; j++) {
      action.Operation(val);
      if (j > i) *tcp = val;
      *trp++ = val;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
TMatrixFSym &TMatrixFSym::Apply(const TElementPosActionF &action)
{ 
  // Apply action to each element of the matrix. To action the location
  // of the current element is passed.
  
  Assert(IsValid());

  Float_t val = 0;
  Float_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
  Float_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
  for (Int_t i = 0; i < fNrows; i++) {
    action.fI = i+fRowLwb;
    trp += i;         // point to [i,i]
    tcp += i*fNcols;  // point to [i,i]
    for (Int_t j = i; j < fNcols; j++) {
      action.fJ = j+fColLwb;
      action.Operation(val);
      if (j > i) *tcp = val;
      *trp++ = val;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
void TMatrixFSym::Randomize(Float_t alpha,Float_t beta,Double_t &seed)
{
  // randomize matrix element values but keep matrix symmetric

  Assert(IsValid());
    
  if (fNrows != fNcols || fRowLwb != fColLwb) {
    Error("Randomize(Float_t,Float_t,Double_t &","matrix should be square");
    return;
  }
  
  const Float_t scale = beta-alpha;
  const Float_t shift = alpha/scale;

  Float_t *ep = GetMatrixArray();
  for (Int_t i = 0; i < fNrows; i++) {
    const Int_t off = i*fNcols;
    for (Int_t j = 0; j <= i; j++) {
      ep[off+j] = scale*(Drand(seed)+shift);
      if (i != j) {
        ep[j*fNcols+i] = ep[off+j];
      }
    }
  }
}

//______________________________________________________________________________
void TMatrixFSym::RandomizePD(Float_t alpha,Float_t beta,Double_t &seed)
{
  // randomize matrix element values but keep matrix symmetric positive definite

  Assert(IsValid());

  if (fNrows != fNcols || fRowLwb != fColLwb) {
    Error("RandomizeSym(Float_t,Float_t,Double_t &","matrix should be square");
    return;
  }

  const Float_t scale = beta-alpha;
  const Float_t shift = alpha/scale;

  Float_t *ep = GetMatrixArray();
  Int_t i;
  for (i = 0; i < fNrows; i++) {
    const Int_t off = i*fNcols;
    for (Int_t j = 0; j <= i; j++)
      ep[off+j] = scale*(Drand(seed)+shift);
  }

  for (i = fNrows-1; i >= 0; i--) {
    const Int_t off1 = i*fNcols;
    for (Int_t j = i; j >= 0; j--) {
      const Int_t off2 = j*fNcols;
      ep[off1+j] *= ep[off2+j];
      for (Int_t k = j-1; k >= 0; k--) {
        ep[off1+j] += ep[off1+k]*ep[off2+k];
      }
      if (i != j)
        ep[off2+i] = ep[off1+j];
    }
  }
}

//______________________________________________________________________________
const TMatrixF TMatrixFSym::EigenVectors(TVectorF &eigenValues) const
{
  // Return a matrix containing the eigen-vectors ordered by descending eigen-values.
  // For full functionality use TMatrixDSymEigen .

  TMatrixDSymEigen eigen(*this);
  eigenValues = eigen.GetEigenValues();
  return eigen.GetEigenVectors();
}

//______________________________________________________________________________
TMatrixFSym operator+(const TMatrixFSym &source1,const TMatrixFSym &source2)
{
  TMatrixFSym target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixFSym operator-(const TMatrixFSym &source1,const TMatrixFSym &source2)
{
  TMatrixFSym target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixFSym operator*(Float_t val,const TMatrixFSym &source)
{
  TMatrixFSym target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixFSym operator*(const TMatrixFSym &source,Float_t val)
{
  TMatrixFSym target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixFSym &Add(TMatrixFSym &target,Float_t scalar,const TMatrixFSym &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    ::Error("Add","matrices not compatible");
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
      *trp++ += tmp;
      tcp += ncols;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixFSym &ElementMult(TMatrixFSym &target,const TMatrixFSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementMult","matrices not compatible");
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
TMatrixFSym &ElementDiv(TMatrixFSym &target,const TMatrixFSym &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    ::Error("ElementDiv","matrices not compatible");
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
void TMatrixFSym::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixFSym.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      Clear();
      TMatrixFBase::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      fElements = new Float_t[fNelems];
      Int_t i;
      for (i = 0; i < fNrows; i++) {
        R__b.ReadFastArray(fElements+i*fNcols+i,fNcols-i);
      }

      if (fNelems <= kSizeMax) {
        memcpy(fDataStack,fElements,fNelems*sizeof(Float_t));
        delete [] fElements;
        fElements = fDataStack;
      }

      // copy to Lower left triangle
      for (i = 0; i < fNrows; i++) {
        for (Int_t j = 0; j < i; j++) {
          fElements[i*fNcols+j] = fElements[j*fNrows+i];
        }
      }
      return;
  } else {
    TMatrixFBase::Class()->WriteBuffer(R__b,this);
    // Only write the Upper right triangle
    for (Int_t i = 0; i < fNrows; i++) {
      R__b.WriteFastArray(fElements+i*fNcols+i,fNcols-i);
    }
  }
}
