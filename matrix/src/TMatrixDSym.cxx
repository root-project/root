// @(#)root/matrix:$Name:  $:$Id: TMatrixDSym.cxx,v 1.3 2004/01/27 08:12:26 brun Exp $
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
// TMatrixDSym                                                          //
//                                                                      //
// Implementation of a symmetric matrix in the linear algebra package   //
//                                                                      //
// Note that in this implementation both matrix element m[i][j] and     //
// m[j][i] are updated and stored in memory . However, when making the  //
// object persistent only the upper right triangle is stored .          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDSym.h"
#include "TDecompLU.h"

ClassImp(TMatrixDSym)

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(Int_t no_rows)
{
  Allocate(no_rows,no_rows,0,0,1);
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(Int_t row_lwb,Int_t row_upb)
{
  const Int_t no_rows = row_upb-row_lwb+1;
  Allocate(no_rows,no_rows,row_lwb,row_lwb,1);
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(Int_t no_rows,const Double_t *elements,Option_t *option)
{
  // option="F": array elements contains the matrix stored column-wise
  //             like in Fortran, so a[i,j] = elements[i+no_rows*j],
  // else        it is supposed that array elements are stored row-wise
  //             a[i,j] = elements[i*no_cols+j]

  Allocate(no_rows,no_rows);
  SetMatrixArray(elements,option);
  if (!IsSymmetric()) {
    Error("TMatrixDSym(Int_t,Double_t*,Option_t*)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(Int_t row_lwb,Int_t row_upb,const Double_t *elements,Option_t *option)
{
  const Int_t no_rows = row_upb-row_lwb+1;
  Allocate(no_rows,no_rows,row_lwb,row_lwb);
  SetMatrixArray(elements,option);
  if (!IsSymmetric()) {
    Error("TMatrixDSym(Int_t,Int_t,Double_t*,Option_t*)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(const TMatrixDSym &another) : TMatrixDBase()
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(const TMatrixFSym &another) : TMatrixDBase()
{
  Assert(another.IsValid());
  Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
  *this = another;
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixDSym &prototype)
{
  // Create a matrix applying a specific operation to the prototype.
  // Example: TMatrixDSym a(10,12); ...; TMatrixDSym b(TMatrixDBase::kTransposed, a);
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
      Error("TMatrixDSym(EMatrixCreatorOp1,const TMatrixDSym)",
             "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(EMatrixCreatorsOp1 op,const TMatrixD &prototype)
{
  Assert(dynamic_cast<TMatrixD *>(this) != &prototype);
  Invalidate();
                   
  Assert(prototype.IsValid());

  switch(op) {
    case kAtA:
      AtMultA(prototype);
      break;

    default:
      Error("TMatrixDSym(EMatrixCreatorOp1,const TMatrixD)",
             "operation %d not yet implemented", op);
  }
}

//______________________________________________________________________________
TMatrixDSym::TMatrixDSym(const TMatrixDSymLazy &lazy_constructor)
{
  Allocate(lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetRowUpb()-lazy_constructor.GetRowLwb()+1,
           lazy_constructor.GetRowLwb(),lazy_constructor.GetRowLwb(),1);
  lazy_constructor.FillIn(*this);
  if (!IsSymmetric()) {
    Error("TMatrixDSym(TMatrixDSymLazy)","matrix not symmetric");
    Invalidate();
  }
}

//______________________________________________________________________________
void TMatrixDSym::Allocate(Int_t no_rows,Int_t no_cols,Int_t row_lwb,Int_t col_lwb,Int_t init)
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
    memset(fElements,0,fNelems*sizeof(Double_t));
}

//______________________________________________________________________________
void TMatrixDSym::AtMultA(const TMatrixD &a,Int_t constr)
{
  // Create a matrix C such that C = A' * A. In other words,
  // c[i,j] = SUM{ a[k,i] * a[k,j] }. Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());

  if (constr)
    Allocate(a.GetNcols(),a.GetNcols(),a.GetColLwb(),a.GetColLwb(),1);

#ifdef CBLAS
  const Double_t *ap = a.GetMatrixArray();
        Double_t *cp = this->GetMatrixArray();
  cblas_dgemm (CblasRowMajor,CblasTrans,CblasNoTrans,fNrows,fNcols,a.GetNrows(),
               1.0,ap,a.GetNcols(),ap,a.GetNcols(),1.0,cp,fNcols);
#else
  const Int_t nb     = a.GetNoElements();
  const Int_t ncolsa = a.GetNcols();
  const Int_t ncolsb = ncolsa;
  const Double_t * const ap = a.GetMatrixArray();
  const Double_t * const bp = ap;
        Double_t *       cp = this->GetMatrixArray();

  const Double_t *acp0 = ap;           // Pointer to  A[i,0];
  while (acp0 < ap+a.GetNcols()) {
    for (const Double_t *bcp = bp; bcp < bp+ncolsb; ) { // Pointer to the j-th column of A, Start bcp = A[0,0]
      const Double_t *acp = acp0;                       // Pointer to the i-th column of A, reset to A[0,i]
      Double_t cij = 0;
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
void TMatrixDSym::AtMultA(const TMatrixDSym &a,Int_t constr)
{
  // Matrix multiplication, with A symmetric
  // Create a matrix C such that C = A' * A = A * A = A * A'
  // Note, matrix C is allocated for constr=1.

  Assert(a.IsValid());

  if (constr)
    Allocate(a.GetNcols(),a.GetNcols(),a.GetColLwb(),a.GetColLwb(),1);

  const Double_t *ap1 = a.GetMatrixArray();
  const Double_t *bp1 = ap1;
        Double_t *cp1 = this->GetMatrixArray();

#ifdef CBLAS
  cblas_dsymm (CblasRowMajor,CblasLeft,CblasUpper,fNrows,fNcols,1.0,
               ap1,a.GetNcols(),bp1,a.GetNcols(),0.0,cp1,fNcols);
#else
  const Double_t *ap2 = a.GetMatrixArray();
  const Double_t *bp2 = ap2;
        Double_t *cp2 = this->GetMatrixArray();
  for (Int_t i = 0; i < fNrows; i++) {
    for (Int_t j = 0; j < fNcols; j++) {
      const Double_t b_ij = *bp1++;
      *cp1 += b_ij*(*ap1);
      Double_t tmp = 0.0;
      ap2 = ap1+1;
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
void TMatrixDSym::Adopt(Int_t nrows,Double_t *data)
{
  if (nrows <= 0)
  {
    Error("Adopt","nrows=%d",nrows);
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
void TMatrixDSym::Adopt(Int_t row_lwb,Int_t row_upb,Double_t *data)
{
  if (row_upb < row_lwb)
  {
    Error("Adopt","row_upb=%d < row_lwb=%d",row_upb,row_lwb);
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
TMatrixDSym TMatrixDSym::GetSub(Int_t row_lwb,Int_t row_upb,Option_t *option) const
{
  // Get submatrix [row_lwb..row_upb][row_lwb..row_upb]; The indexing range of the
  // returned matrix depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1][0..row_upb-row_lwb+1] (default)
  // else          : return [row_lwb..row_upb][row_lwb..row_upb]

  Assert(IsValid());

  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    return TMatrixDSym();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    return TMatrixDSym();
  }
  if (row_upb < row_lwb) {
    Error("GetSub","row_upb < row_lwb");
    return TMatrixDSym();
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

  TMatrixDSym sub(row_lwb_sub,row_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;

  const Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNrows+(row_lwb-fRowLwb);
        Double_t *bp = sub.GetMatrixArray();

  for (Int_t irow = 0; irow < nrows_sub; irow++) {
    const Double_t *ap_sub = ap;
    for (Int_t icol = 0; icol < nrows_sub; icol++) {
      *bp++ = *ap_sub++;
    }
    ap += fNrows;
  }

  return sub;
}

//______________________________________________________________________________
void TMatrixDSym::SetSub(Int_t row_lwb,const TMatrixDSym &source)
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

  const Double_t *bp = source.GetMatrixArray();
        Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb)*fNrows+(row_lwb-fRowLwb);

  for (Int_t irow = 0; irow < nRows_source; irow++) {
    Double_t *ap_sub = ap;
    for (Int_t icol = 0; icol < nRows_source; icol++) {
      *ap_sub++ = *bp++;
    }
    ap += fNrows;
  }
}

//______________________________________________________________________________
Double_t TMatrixDSym::Determinant() const
{
  const TMatrixD &tmp = *this;
  TDecompLU lu(tmp,fTol);
  Double_t d1,d2;
  lu.Det(d1,d2);
  return d1*TMath::Power(2.0,d2);
}

//______________________________________________________________________________
void TMatrixDSym::Determinant(Double_t &d1,Double_t &d2) const
{
  const TMatrixD &tmp = *this;
  TDecompLU lu(tmp,fTol);
  lu.Det(d1,d2);
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Zero()
{
  Assert(IsValid());
  memset(this->GetMatrixArray(),0,fNelems*sizeof(Double_t));

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Abs()
{
  // Take an absolute value of a matrix, i.e. apply Abs() to each element.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = TMath::Abs(*ep);

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Sqr()
{
  // Square each element of the matrix.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = (*ep) * (*ep);

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Sqrt()
{
  // Take square root of all elements.

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last) {
    Assert(*ep >= 0);
    *ep++ = TMath::Sqrt(*ep);
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::UnitMatrix()
{
  // Make a unit matrix (matrix need not be a square one).

  Assert(IsValid());

  Double_t *ep = this->GetMatrixArray();
  const Double_t * const ep_last = ep+fNelems;
  memset(ep,0,fNelems*sizeof(Double_t));
  while (ep < ep_last) {
    *ep = 1.0;
    ep += fNcols+1;
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Transpose(const TMatrixDSym &source)
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
TMatrixDSym &TMatrixDSym::NormByDiag(const TVectorD &v,Option_t *option)
{
  // b(i,j) = a(i,j)/sqrt(abs*(v(i)*v(j)))

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

  const Double_t *pV = v.GetMatrixArray();
        Double_t *mp = this->GetMatrixArray();

  if (divide) {
    for (Int_t irow = 0; irow < fNrows; irow++) {
      for (Int_t icol = 0; icol < fNcols; icol++) {
        const Double_t val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
        Assert(val != 0.0);
        *mp++ /= val;
      }
    }
  } else {
    for (Int_t irow = 0; irow < fNrows; irow++) {
      for (Int_t icol = 0; icol < fNcols; icol++) {
        const Double_t val = TMath::Sqrt(TMath::Abs(pV[irow]*pV[icol]));
        *mp++ *= val;
      }
    }
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator=(const TMatrixDSym &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=","matrices not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(this->GetMatrixArray(),source.fElements,fNelems*sizeof(Double_t));
  }
  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator=(const TMatrixDSymLazy &lazy_constructor)
{
  Assert(IsValid());

  if (lazy_constructor.fRowUpb != GetRowUpb() ||
      lazy_constructor.fRowLwb != GetRowLwb()) {
     Error("operator=(const TMatrixDSymLazy&)", "matrix is incompatible with "
           "the assigned Lazy matrix");
    Invalidate();
    return *this;
  }

  lazy_constructor.FillIn(*this);
  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator=(Double_t val)
{
  // Assign val to every element of the matrix.

  Assert(IsValid());

  Double_t *ep = fElements;
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator+=(Double_t val)
{
  // Add val to every element of the matrix.

  Assert(IsValid());

  Double_t *ep = fElements;
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator-=(Double_t val)
{
  // Subtract val from every element of the matrix.

  Assert(IsValid());

  Double_t *ep = fElements;
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator*=(Double_t val)
{
  // Multiply every element of the matrix with val.

  Assert(IsValid());

  Double_t *ep = fElements;
  const Double_t * const ep_last = ep+fNelems;
  while (ep < ep_last)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::operator+=(const TMatrixDSym &source)
{
  // Add the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator+=","matrices not compatible");
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
TMatrixDSym &TMatrixDSym::operator-=(const TMatrixDSym &source)
{
  // Subtract the source matrix.

  if (!AreCompatible(*this,source)) {
    Error("operator-=","matrices not compatible");
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
      if (j > i) *tcp -= *sp;
      *trp++ -= *sp++;
      tcp += fNcols;
    }
    tcp -= fNelems-1; // point to [0,i]
  }

  return *this;
}

//______________________________________________________________________________
TMatrixDSym &TMatrixDSym::Apply(const TElementActionD &action)
{ 
  Assert(IsValid());
  
  Double_t val = 0;
  Double_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
  Double_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
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
TMatrixDSym &TMatrixDSym::Apply(const TElementPosActionD &action)
{ 
  // Apply action to each element of the matrix. To action the location
  // of the current element is passed.
  
  Assert(IsValid());

  Double_t val = 0;
  Double_t *trp = this->GetMatrixArray(); // pointer to UR part and diagonal, traverse row-wise
  Double_t *tcp = trp;                 // pointer to LL part,              traverse col-wise
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
Bool_t operator==(const TMatrixDSym &m1,const TMatrixDSym &m2)
{
  // Check to see if two matrices are identical.

  if (!AreCompatible(m1,m2)) return kFALSE;
  return (memcmp(m1.GetMatrixArray(),m2.GetMatrixArray(),
                 m1.GetNoElements()*sizeof(Double_t)) == 0);
}

//______________________________________________________________________________
TMatrixDSym operator+(const TMatrixDSym &source1,const TMatrixDSym &source2)
{
  TMatrixDSym target(source1);
  target += source2;
  return target;
}

//______________________________________________________________________________
TMatrixDSym operator-(const TMatrixDSym &source1,const TMatrixDSym &source2)
{
  TMatrixDSym target(source1);
  target -= source2;
  return target;
}

//______________________________________________________________________________
TMatrixDSym operator*(Double_t val,const TMatrixDSym &source)
{
  TMatrixDSym target(source);
  target *= val;
  return target;
}

//______________________________________________________________________________
TMatrixDSym &Add(TMatrixDSym &target,Double_t scalar,const TMatrixDSym &source)
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
      *trp++ += tmp;
      tcp += ncols;
    }
    tcp -= nelems-1; // point to [0,i]
  }

  return target;
}

//______________________________________________________________________________
TMatrixDSym &ElementMult(TMatrixDSym &target,const TMatrixDSym &source)
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
TMatrixDSym &ElementDiv(TMatrixDSym &target,const TMatrixDSym &source)
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
void TMatrixDSym::Streamer(TBuffer &R__b)
{
  // Stream an object of class TMatrixDSym.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      Clear();
      TMatrixDBase::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      fElements = new Double_t[fNelems];
      Int_t i;
      for (i = 0; i < fNrows; i++) {
        R__b.ReadFastArray(fElements+i*fNcols+i,fNcols-i);
      }
      if (fNelems <= kSizeMax) {
        memcpy(fDataStack,fElements,fNelems*sizeof(Double_t));
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
    TMatrixDBase::Class()->WriteBuffer(R__b,this);
    // Only write the Upper right triangle
    for (Int_t i = 0; i < fNrows; i++) {
      R__b.WriteFastArray(fElements+i*fNcols+i,fNcols-i);
    }
  }
}
