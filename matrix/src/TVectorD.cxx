// @(#)root/matrix:$Name:  $:$Id: TVectorD.cxx,v 1.46 2004/05/27 06:39:53 brun Exp $
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
// TVectorD                                                             //
//                                                                      //
// Implementation of Vectors in the linear algebra package              //
//                                                                      //
// Unless otherwise specified, vector indices always start with 0,      //
// spanning up to the specified limit-1.                                //
//                                                                      //
// For (n) vectors where n <= kSizeMax (5 currently) storage space is   //
// available on the stack, thus avoiding expensive allocation/          //
// deallocation of heap space . However, this introduces of course      //
// kSizeMax overhead for each vector object . If this is an issue       //
// recompile with a new appropriate value (>=0) for kSizeMax            //
//                                                                      //
// Another way to assign and store vector data is through Use           //
// see for instance stress_linalg.cxx file .                            //
//                                                                      //
// Note that Constructors/assignments exists for all different matrix   //
// views                                                                //
//                                                                      //
// For usage examples see $ROOTSYS/test/stress_linalg.cxx               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVectorD.h"

ClassImp(TVectorD)

//______________________________________________________________________________
void TVectorD::Delete_m(Int_t size,Double_t *&m)
{
  if (m) {
    if (size > kSizeMax)
    {
      delete [] m;
      m = 0;
    }
  }
}

//______________________________________________________________________________
Double_t* TVectorD::New_m(Int_t size)
{
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
Int_t TVectorD::Memcpy_m(Double_t *newp,const Double_t *oldp,Int_t copySize,
                         Int_t newSize,Int_t oldSize)
{
  if (copySize == 0 || oldp == newp) return 0;
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
    else {
      memcpy(newp,oldp,copySize*sizeof(Double_t));
    }
  }
  return 0;
}

//______________________________________________________________________________
void TVectorD::Allocate(Int_t nrows,Int_t row_lwb,Int_t init)
{
  // Allocate new vector. Arguments are number of rows and row
  // lowerbound (0 default).

  Invalidate();
  Assert(nrows > 0);

  SetBit(TMatrixDBase::kStatus);
  fNrows   = nrows;
  fRowLwb  = row_lwb;
  fIsOwner = kTRUE;

  fElements = New_m(fNrows);
  if (init)
    memset(fElements,0,fNrows*sizeof(Double_t));
}

//______________________________________________________________________________
TVectorD::TVectorD(Int_t n)
{
  Allocate(n,0,1);
}

//______________________________________________________________________________
TVectorD::TVectorD(Int_t lwb,Int_t upb)
{
  Allocate(upb-lwb+1,lwb,1);
}

//______________________________________________________________________________
TVectorD::TVectorD(Int_t n,const Double_t *elements)
{
  Allocate(n,0);
  SetElements(elements);
}

//______________________________________________________________________________
TVectorD::TVectorD(Int_t lwb,Int_t upb,const Double_t *elements)
{
  Allocate(upb-lwb+1,lwb);
  SetElements(elements);
}

//______________________________________________________________________________
TVectorD::TVectorD(const TVectorD &another) : TObject(another)
{
  Assert(another.IsValid());
  Allocate(another.GetUpb()-another.GetLwb()+1,another.GetLwb());
  *this = another;
}

//______________________________________________________________________________
TVectorD::TVectorD(const TVectorF &another) : TObject(another)
{
  Assert(another.IsValid());
  Allocate(another.GetUpb()-another.GetLwb()+1,another.GetLwb());
  *this = another;
}

//______________________________________________________________________________
TVectorD::TVectorD(const TMatrixDRow_const &mr) : TObject()
{
  const TMatrixDBase *mt = mr.GetMatrix();
  Assert(mt->IsValid());
  Allocate(mt->GetColUpb()-mt->GetColLwb()+1,mt->GetColLwb());
  *this = mr;
}

//______________________________________________________________________________
TVectorD::TVectorD(const TMatrixDColumn_const &mc) : TObject()
{
  const TMatrixDBase *mt = mc.GetMatrix();
  Assert(mt->IsValid());
  Allocate(mt->GetRowUpb()-mt->GetRowLwb()+1,mt->GetRowLwb());
  *this = mc;
}

//______________________________________________________________________________
TVectorD::TVectorD(const TMatrixDDiag_const &md) : TObject()
{
  const TMatrixDBase *mt = md.GetMatrix();
  Assert(mt->IsValid());
  Allocate(TMath::Min(mt->GetNrows(),mt->GetNcols()));
  *this = md;
}

//______________________________________________________________________________
TVectorD::TVectorD(Int_t lwb,Int_t upb,Double_t va_(iv1), ...)
{
  // Make a vector and assign initial values. Argument list should contain
  // Double_t values to assign to vector elements. The list must be
  // terminated by the string "END". Example:
  // TVectorD foo(1,3,0.0,1.0,1.5,"END");

  const Int_t no_rows = upb-lwb+1;
  Assert(no_rows);
  Allocate(no_rows,lwb);

  va_list args;
  va_start(args,va_(iv1));             // Init 'args' to the beginning of
                                       // the variable length list of args

  (*this)(lwb) = iv1;
  for (Int_t i = lwb+1; i <= upb; i++)
    (*this)(i) = (Double_t)va_arg(args,Double_t);

  if (strcmp((char *)va_arg(args,char *),"END"))
    Error("TVectorD(Int_t, Int_t, ...)", "argument list must be terminated by \"END\"");

  va_end(args);
}

//______________________________________________________________________________
void TVectorD::ResizeTo(Int_t lwb,Int_t upb)
{
  // Resize the vector to [lwb:upb] .
  // New dynamic elemenst are created, the overlapping part of the old ones are
  // copied to the new structures, then the old elements are deleleted.

  Assert(IsValid());
  if (!fIsOwner) {
    Error("ResizeTo(lwb,upb)","Not owner of data array,cannot resize");
    return;
  }

  const Int_t new_nrows = upb-lwb+1;

  if (fNrows > 0) {

    if (fNrows == new_nrows && fRowLwb == lwb)
      return;
    else if (new_nrows == 0) {
      Clear();
      return;
    }

    Double_t    *elements_old = GetMatrixArray();
    const Int_t  nrows_old    = fNrows;
    const Int_t  rowLwb_old   = fRowLwb;

    Allocate(new_nrows,lwb);
    Assert(IsValid());
    if (fNrows > kSizeMax || nrows_old > kSizeMax)
      memset(GetMatrixArray(),0,fNrows*sizeof(Double_t));
    else if (fNrows > nrows_old)
      memset(GetMatrixArray()+nrows_old,0,(fNrows-nrows_old)*sizeof(Double_t));

    // Copy overlap
    const Int_t rowLwb_copy = TMath::Max(fRowLwb,rowLwb_old);
    const Int_t rowUpb_copy = TMath::Min(fRowLwb+fNrows-1,rowLwb_old+nrows_old-1);
    const Int_t nrows_copy  = rowUpb_copy-rowLwb_copy+1;

    const Int_t nelems_new = fNrows;
    Double_t *elements_new = GetMatrixArray();
    if (nrows_copy > 0) {
      const Int_t rowOldOff = rowLwb_copy-rowLwb_old;
      const Int_t rowNewOff = rowLwb_copy-fRowLwb;
      Memcpy_m(elements_new+rowNewOff,elements_old+rowOldOff,nrows_copy,nelems_new,nrows_old);
    }

    Delete_m(nrows_old,elements_old);
  } else {
    Allocate(upb-lwb+1,lwb,1);
  }
}

//______________________________________________________________________________
void TVectorD::Use(Int_t n,Double_t *data)
{
  Assert(n > 0);

  Clear();
  fNrows    = n;
  fRowLwb   = 0;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
void TVectorD::Use(Int_t lwb,Int_t upb,Double_t *data)
{
  Assert(upb >= lwb);

  Clear();
  fNrows    = upb-lwb+1;
  fRowLwb   = lwb;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
TVectorD TVectorD::GetSub(Int_t row_lwb,Int_t row_upb,Option_t *option) const
{
  // Get subvector [row_lwb..row_upb]; The indexing range of the
  // returned vector depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1] (default)
  // else          : return [row_lwb..row_upb]

  Assert(IsValid());
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    return TVectorD();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    return TVectorD();
  }
  if (row_upb < row_lwb) {
    Error("GetSub","row_upb < row_lwb");
    return TVectorD();
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

  TVectorD sub(row_lwb_sub,row_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;

  const Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb);
        Double_t *bp = sub.GetMatrixArray();

  for (Int_t irow = 0; irow < nrows_sub; irow++)
      *bp++ = *ap++;

  return sub;
}

//______________________________________________________________________________
void TVectorD::SetSub(Int_t row_lwb,const TVectorD &source)
{
  // Insert vector source starting at [row_lwb], thereby overwriting the part
  // [row_lwb..row_lwb+nrows_source];

  Assert(IsValid());
  Assert(source.IsValid());

  if (row_lwb < fRowLwb && row_lwb > fRowLwb+fNrows-1) {
    Error("SetSub","row_lwb outof bounds");
    return;
  }
  const Int_t nRows_source = source.GetNrows();
  if (row_lwb+nRows_source > fRowLwb+fNrows) {
    Error("SetSub","source vector too large");
    return;
  }

  const Double_t *bp = source.GetMatrixArray();
        Double_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb);

  for (Int_t irow = 0; irow < nRows_source; irow++)
    *ap++ = *bp++;
}

//______________________________________________________________________________
TVectorD &TVectorD::Zero()
{
  Assert(IsValid());
  memset(this->GetMatrixArray(),0,fNrows*sizeof(Double_t));
  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::Abs()
{
  // Take an absolute value of a vector, i.e. apply Abs() to each element.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep = TMath::Abs(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::Sqr()
{
  // Square each element of the vector.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep = (*ep) * (*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::Sqrt()
{
   // Take square root of all elements.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    Assert(*ep >= 0);
    *ep = TMath::Sqrt(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::Invert()
{
   // v[i] = 1/v[i]

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows; 
  while (ep < fp) {
    Assert(*ep != 0.0);
    *ep = 1./ *ep;
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::SelectNonZeros(const TVectorD &select)
{   
  if (!AreCompatible(*this,select)) {
    Error("SelectNonZeros(const TVectorD &","vector's not compatible");
    Invalidate();
    return *this;
  }
  
  const Double_t *sp = select.GetMatrixArray();
        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows; 
  while (ep < fp) {
    if (*sp == 0.0)
      *ep = 0.0;
    sp++; ep++;
  } 

  return *this;
}   

//______________________________________________________________________________
Double_t TVectorD::Norm1() const
{
  // Compute the 1-norm of the vector SUM{ |v[i]| }.

  Assert(IsValid());

  Double_t norm = 0;
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    norm += TMath::Abs(*ep++);

  return norm;
}

//______________________________________________________________________________
Double_t TVectorD::Norm2Sqr() const
{
  // Compute the square of the 2-norm SUM{ v[i]^2 }.

  Assert(IsValid());

  Double_t norm = 0;
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    norm += (*ep) * (*ep);
    ep++;
  }

  return norm;
}

//______________________________________________________________________________
Double_t TVectorD::NormInf() const
{
  // Compute the infinity-norm of the vector MAX{ |v[i]| }.

  Assert(IsValid());

  Double_t norm = 0;
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    norm = TMath::Max(norm,TMath::Abs(*ep++));

  return norm;
}

//______________________________________________________________________________
Int_t TVectorD::NonZeros() const
{
  // Compute the number of elements != 0.0

  Assert(IsValid());

  Int_t nr_nonzeros = 0;
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (*ep++) nr_nonzeros++;

  return nr_nonzeros;
}

//______________________________________________________________________________
Double_t TVectorD::Sum() const
{
  // Compute sum of elements

  Assert(IsValid());

  Double_t sum = 0.0;
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) 
    sum += *ep++;

  return sum;
}

//______________________________________________________________________________
Double_t TVectorD::Min() const
{
  // return minimum vector element value

  Assert(IsValid());

  const Int_t index = TMath::LocMin(fNrows,fElements);
  return fElements[index];
}

//______________________________________________________________________________
Double_t TVectorD::Max() const
{
  // return maximum vector element value

  Assert(IsValid());

  const Int_t index = TMath::LocMax(fNrows,fElements);
  return fElements[index];
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TVectorD &source)
{
  // Notice that this assignment does NOT change the ownership :
  // if the storage space was adopted, source is copied to
  // this space .

  if (!AreCompatible(*this,source)) {
    Error("operator=(const TVectorD &)","vectors not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNrows*sizeof(Double_t));
  }
  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TVectorF &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TVectorF &)","vectors not compatible");
    Invalidate();
    return *this;
  }

  if (dynamic_cast<TVectorF *>(this) != &source) {
    TObject::operator=(source);
    const Float_t  * const ps = source.GetMatrixArray();
          Double_t * const pt = GetMatrixArray();
    for (Int_t i = 0; i < fNrows; i++)
      pt[i] = (Double_t) ps[i];
  }
  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TMatrixDRow_const &mr)
{
  // Assign a matrix row to a vector.

  Assert(IsValid());
  const TMatrixDBase *mt = mr.GetMatrix();
  Assert(mt->IsValid());

  if (mt->GetColLwb() != fRowLwb || mt->GetNcols() != fNrows) {
    Error("operator=(const TMatrixDRow_const &)","vector and row not compatible");
    Invalidate();
    return *this;
  }

  const Int_t inc    = mr.GetInc();
  const Double_t *rp = mr.GetPtr();              // Row ptr
        Double_t *ep = this->GetMatrixArray();   // Vector ptr
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *rp;
     rp += inc;
  }

  Assert(rp == mr.GetPtr()+mt->GetNcols());

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TMatrixDColumn_const &mc)
{
   // Assign a matrix column to a vector.

  Assert(IsValid());
  const TMatrixDBase *mt = mc.GetMatrix();
  Assert(mt->IsValid());

  if (mt->GetRowLwb() != fRowLwb || mt->GetNrows() != fNrows) {
    Error("operator=(const TMatrixDColumn_const &)","vector and column not compatible");
    Invalidate();
    return *this;
  }

  const Int_t inc    = mc.GetInc();
  const Double_t *cp = mc.GetPtr();              // Column ptr
        Double_t *ep = this->GetMatrixArray();   // Vector ptr
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *cp;
     cp += inc;
  }

  Assert(cp == mc.GetPtr()+mt->GetNoElements());

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TMatrixDDiag_const &md)
{
  // Assign the matrix diagonal to a vector.

  Assert(IsValid());
  const TMatrixDBase *mt = md.GetMatrix();
  Assert(mt->IsValid());

  if (md.GetNdiags() != fNrows) {
    Error("operator=(const TMatrixDDiag_const &)","vector and matrix-diagonal not compatible");
    Invalidate();
    return *this;
  }

  const Int_t    inc = md.GetInc();
  const Double_t *dp = md.GetPtr();              // Diag ptr
        Double_t *ep = this->GetMatrixArray();   // Vector ptr
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *dp;
     dp += inc;
  }

  Assert(dp < md.GetPtr()+mt->GetNoElements()+inc);

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TMatrixDSparseRow_const &mr)
{
  // Assign a sparse matrix row to a vector. The matrix row is implicitly transposed
  // to allow the assignment in the strict sense.

  Assert(IsValid());
  const TMatrixDBase *mt = mr.GetMatrix();
  Assert(mt->IsValid());

  if (mt->GetColLwb() != fRowLwb || mt->GetNcols() != fNrows) {
    Error("operator=(const TMatrixDSparseRow_const &)","vector and row not compatible");
    Invalidate();
    return *this;
  }

  const Int_t nIndex = mr.GetNindex();
  const Double_t * const prData = mr.GetDataPtr();          // Row Data ptr
  const Int_t    * const prCol  = mr.GetColPtr();           // Col ptr
        Double_t * const pvData = this->GetMatrixArray();   // Vector ptr

  memset(pvData,0,fNrows*sizeof(Double_t));
  for (Int_t index = 0; index < nIndex; index++) {
    const Int_t icol = prCol[index];
    pvData[icol] = prData[index];
  }

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(const TMatrixDSparseDiag_const &md)
{
  // Assign a sparse matrix diagonal to a vector.

  Assert(IsValid());
  const TMatrixDBase *mt = md.GetMatrix();
  Assert(mt->IsValid());

  if (md.GetNdiags() != fNrows) {
    Error("operator=(const TMatrixDSparseDiag_const &)","vector and matrix-diagonal not compatible");
    Invalidate();
    return *this;
  } 

  Double_t * const pvData = this->GetMatrixArray();
  for (Int_t idiag = 0; idiag < fNrows; idiag++)
    pvData[idiag] = md(idiag);

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator=(Double_t val)
{
  // Assign val to every element of the vector.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator+=(Double_t val)
{
  // Add val to every element of the vector.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator-=(Double_t val)
{
  // Subtract val from every element of the vector.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator*=(Double_t val)
{
  // Multiply every element of the vector with val.

  Assert(IsValid());

        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator+=(const TVectorD &source)
{
  // Add vector source

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TVectorD &)","vector's not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp = source.GetMatrixArray();
        Double_t *tp = this->GetMatrixArray();
  const Double_t * const tp_last = tp+fNrows;
  while (tp < tp_last)
    *tp++ += *sp++;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator-=(const TVectorD &source)
{
  // Subtract vector source

  if (!AreCompatible(*this,source)) {
    Error("operator-=(const TVectorD &)","vector's not compatible");
    Invalidate();
    return *this;
  }

  const Double_t *sp = source.GetMatrixArray();
        Double_t *tp = this->GetMatrixArray();
  const Double_t * const tp_last = tp+fNrows;
  while (tp < tp_last)
    *tp++ -= *sp++;

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator*=(const TMatrixD &a)
{
  // "Inplace" multiplication target = A*target. A needn't be a square one
  // If target has to be resized, it should own the storage: fIsOwner = kTRUE

  Assert(IsValid());
  Assert(a.IsValid());

  if (a.GetNcols() != fNrows || a.GetColLwb() != fRowLwb) {
    Error("operator*=(const TMatrixD &)","vector and matrix incompatible");
    Invalidate();
    return *this;
  }

  if ((fNrows != a.GetNrows() || fRowLwb != a.GetRowLwb()) && !fIsOwner) {
    Error("operator*=(const TMatrixD &)","vector has to be resized but not owner");
    Invalidate();
    return *this;
  }

  const Int_t nrows_old = fNrows;
  Double_t *elements_old;
  if (nrows_old <= kSizeMax) {
    elements_old = new Double_t[nrows_old];
    memcpy(elements_old,fElements,nrows_old*sizeof(Double_t));
  }
  else
    elements_old = fElements;

  fRowLwb = a.GetRowLwb();
  Assert((fNrows = a.GetNrows()) > 0);

  Allocate(fNrows,fRowLwb);

  const Double_t *mp = a.GetMatrixArray();     // Matrix row ptr
        Double_t *tp = this->GetMatrixArray(); // Target vector ptr
#ifdef CBLAS
  cblas_dgemv(CblasRowMajor,CblasNoTrans,a.GetNrows(),a.GetNcols(),1.0,mp,
              a.GetNcols(),elements_old,1,0.0,tp,1);
#else
  const Double_t * const tp_last = tp+fNrows;
  while (tp < tp_last) {
    Double_t sum = 0;
    for (const Double_t *sp = elements_old; sp < elements_old+nrows_old; )
      sum += *sp++ * *mp++;
    *tp++ = sum;
  }
  Assert(mp == a.GetMatrixArray()+a.GetNoElements());
#endif

  if (nrows_old <= kSizeMax)
    delete [] elements_old;
  else
    Delete_m(nrows_old,elements_old);

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator*=(const TMatrixDSparse &a)
{
  // "Inplace" multiplication target = A*target. A needn't be a square one
  // If target has to be resized, it should own the storage: fIsOwner = kTRUE

  Assert(IsValid());
  Assert(a.IsValid());

  if (a.GetNcols() != fNrows || a.GetColLwb() != fRowLwb) {
    Error("operator*=(const TMatrixDSparse &)","vector and matrix incompatible");
    Invalidate();
    return *this;
  }

  if ((fNrows != a.GetNrows() || fRowLwb != a.GetRowLwb()) && !fIsOwner) {
    Error("operator*=(const TMatrixDSparse &)","vector has to be resized but not owner");
    Invalidate();
    return *this;
  }

  const Int_t nrows_old = fNrows;
  Double_t *elements_old;
  if (nrows_old <= kSizeMax) {
    elements_old = new Double_t[nrows_old];
    memcpy(elements_old,fElements,nrows_old*sizeof(Double_t));
  }
  else
    elements_old = fElements;

  fRowLwb = a.GetRowLwb();
  Assert((fNrows = a.GetNrows()) > 0);

  Allocate(fNrows,fRowLwb);

  const Int_t    * const pRowIndex = a.GetRowIndexArray();
  const Int_t    * const pColIndex = a.GetColIndexArray();
  const Double_t * const mp        = a.GetMatrixArray();     // Matrix row ptr

  const Double_t * const sp = elements_old;
        Double_t *       tp = this->GetMatrixArray(); // Target vector ptr

  for (Int_t irow = 0; irow < fNrows; irow++) {
    const Int_t sIndex = pRowIndex[irow]; 
    const Int_t eIndex = pRowIndex[irow+1];
    Double_t sum = 0.0;
    for (Int_t index = sIndex; index < eIndex; index++) {
      const Int_t icol = pColIndex[index];
      sum += mp[index]*sp[icol];
    }
    tp[irow] = sum;
  }

  if (nrows_old <= kSizeMax)
    delete [] elements_old;
  else
    Delete_m(nrows_old,elements_old);

  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::operator*=(const TMatrixDSym &a)
{
  // "Inplace" multiplication target = A*target. A is symmetric .
  // vector size will not change

  Assert(IsValid());
  Assert(a.IsValid());

  if (a.GetNcols() != fNrows || a.GetColLwb() != fRowLwb) {
    Error("operator*=(const TMatrixDSym &)","vector and matrix incompatible");
    Invalidate();
    return *this;
  }

  Double_t * const elements_old = new Double_t[fNrows];
  memcpy(elements_old,fElements,fNrows*sizeof(Double_t));
  memset(fElements,0,fNrows*sizeof(Double_t));

  const Double_t *mp1 = a.GetMatrixArray(); // Matrix row ptr
        Double_t *tp1 = fElements;          // Target vector ptr
#ifdef CBLAS
  cblas_dsymv(CblasRowMajor,CblasUpper,fNrows,1.0,mp1,
              fNrows,elements_old,1,0.0,tp1,1);
#else
  const Double_t *mp2;
  const Double_t *sp1 = elements_old;
  const Double_t *sp2 = sp1;
        Double_t *tp2 = tp1;       // Target vector ptr

  for (Int_t i = 0; i < fNrows; i++) {
    Double_t vec_i = *sp1++;
    *tp1 += *mp1 * vec_i;
    Double_t tmp = 0.0;
    mp2 = mp1+1;
    sp2 = sp1;
    tp2 = tp1+1;
    for (Int_t j = i+1; j < fNrows; j++) {
      const Double_t a_ij = *mp2++;
      *tp2++ += a_ij * vec_i;
      tmp += a_ij * *sp2++;
    }
    *tp1++ += tmp;
    mp1 += fNrows+1;
  }

  Assert(tp1 == fElements+fNrows);
#endif

  delete [] elements_old;

  return *this;
}

//______________________________________________________________________________
Bool_t TVectorD::operator==(Double_t val) const
{
  // Are all vector elements equal to val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ == val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::operator!=(Double_t val) const
{
  // Are all vector elements not equal to val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ != val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::operator<(Double_t val) const
{
  // Are all vector elements < val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ < val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::operator<=(Double_t val) const
{
  // Are all vector elements <= val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ <= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::operator>(Double_t val) const
{
  // Are all vector elements > val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ > val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::operator>=(Double_t val) const
{
  // Are all vector elements >= val?

  Assert(IsValid());

  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ >= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::MatchesNonZeroPattern(const TVectorD &select)
{
  if (!AreCompatible(*this,select)) {
    Error("MatchesNonZeroPattern(const TVectorD&)","vector's not compatible");
    return kFALSE;
  } 

  const Double_t *sp = select.GetMatrixArray();
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    if (*sp == 0.0 && *ep != 0.0)
      return kFALSE;
    sp++; ep++;
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorD::SomePositive(const TVectorD &select)
{
  if (!AreCompatible(*this,select)) {
    Error("SomePositive(const TVectorD&)","vector's not compatible");
    return kFALSE;
  }

  const Double_t *sp = select.GetMatrixArray();
  const Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    if (*sp != 0.0 && *ep <= 0.0)
      return kFALSE;
    sp++; ep++;
  }

  return kTRUE;
}

//______________________________________________________________________________
void TVectorD::AddSomeConstant(Double_t val,const TVectorD &select)
{
  if (!AreCompatible(*this,select))
    Error("AddSomeConstant(Double_t,const TVectorD &)(const TVectorD&)","vector's not compatible");

  const Double_t *sp = select.GetMatrixArray();
        Double_t *ep = this->GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp) {
    if (*sp)
      *ep += val;
    sp++; ep++;
  }
}

extern Double_t Drand(Double_t &ix);

//______________________________________________________________________________
void TVectorD::Randomize(Double_t alpha,Double_t beta,Double_t &seed)
{
  // randomize vector elements value
   
  Assert(IsValid());
   
  const Double_t scale = beta-alpha;
  const Double_t shift = alpha/scale;

        Double_t *       ep = GetMatrixArray();
  const Double_t * const fp = ep+fNrows;
  while (ep < fp)  
    *ep++ = scale*(Drand(seed)+shift);
}

//______________________________________________________________________________
TVectorD &TVectorD::Apply(const TElementActionD &action)
{
  // Apply action to each element of the vector.

  Assert(IsValid());
  for (Double_t *ep = fElements; ep < fElements+fNrows; ep++)
    action.Operation(*ep);
  return *this;
}

//______________________________________________________________________________
TVectorD &TVectorD::Apply(const TElementPosActionD &action)
{
  // Apply action to each element of the vector. In action the location
  // of the current element is known.

  Assert(IsValid());

  Double_t *ep = fElements;
  for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
    action.Operation(*ep++);

  Assert(ep == fElements+fNrows);

  return *this;
}

//______________________________________________________________________________
void TVectorD::Draw(Option_t *option)
{
  // Draw this vector using an intermediate histogram
  // The histogram is named "TVectorD" by default and no title

  //create the hist utility manager (a plugin)
  TVirtualUtilHist *util = (TVirtualUtilHist*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilHist");
  if (!util) {
    TPluginHandler *h;
    if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilHist"))) {
      if (h->LoadPlugin() == -1)
        return;
      h->ExecPlugin(0);
      util = (TVirtualUtilHist*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilHist");
    }
  }
  util->PaintVector(*this,option);
}

//______________________________________________________________________________
void TVectorD::Print(Option_t *flag) const
{
  // Print the vector as a list of elements.

  Assert(IsValid());

  printf("\nVector (%d) %s is as follows",fNrows,flag);

  printf("\n\n     |   %6d  |", 1);
  printf("\n%s\n", "------------------");
  for (Int_t i = 0; i < fNrows; i++) {
    printf("%4d |",i+fRowLwb);
    //printf("%11.4g \n",(*this)(i+fRowLwb));
    printf("%g \n",(*this)(i+fRowLwb));
  }
  printf("\n");
}

//______________________________________________________________________________
Bool_t operator==(const TVectorD &v1,const TVectorD &v2)
{
  // Check to see if two vectors are identical.

  if (!AreCompatible(v1,v2)) return kFALSE;
  return (memcmp(v1.GetMatrixArray(),v2.GetMatrixArray(),v1.GetNrows()*sizeof(Double_t)) == 0);
}

//______________________________________________________________________________
Double_t operator*(const TVectorD &v1,const TVectorD &v2)
{
  // Compute the scalar product.

  if (!AreCompatible(v1,v2)) {
    Error("operator*(const TVectorD &,const TVectorD &)","vector's are incompatible");
    return 0.0;
  }

  const Double_t *v1p = v1.GetMatrixArray();
  const Double_t *v2p = v2.GetMatrixArray();

  Double_t sum = 0.0;
  const Double_t * const fv1p = v1p+v1.GetNrows();
  while (v1p < fv1p)
    sum += *v1p++ * *v2p++;

  return sum;
}

//______________________________________________________________________________
TVectorD operator+(const TVectorD &source1,const TVectorD &source2)
{
  TVectorD target = source1;
  target += source2;
  return target;
}

//______________________________________________________________________________
TVectorD operator-(const TVectorD &source1,const TVectorD &source2)
{
  TVectorD target = source1;
  target -= source2;
  return target;
}

//______________________________________________________________________________
TVectorD operator*(const TMatrixD &a,const TVectorD &source)
{
  TVectorD target = source;
  target *= a;
  return target;
}

//______________________________________________________________________________
TVectorD operator*(const TMatrixDSym &a,const TVectorD &source)
{
  TVectorD target = source;
  target *= a;
  return target;
}

//______________________________________________________________________________
TVectorD operator*(const TMatrixDSparse &a,const TVectorD &source)
{
  TVectorD target = source;
  target *= a;
  return target;
}

//______________________________________________________________________________
TVectorD operator*(Double_t val,const TVectorD &source)
{
  TVectorD target = source;
  target *= val;
  return target;
}

//______________________________________________________________________________
TVectorD &Add(TVectorD &target,Double_t scalar,const TVectorD &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    Error("Add(TVectorD &,Double_t,const TVectorD &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp  = source.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();
  if (scalar == 1.0 ) {
    while ( tp < ftp )
      *tp++ += *sp++;
  } else if (scalar == -1.0) {
    while ( tp < ftp )
      *tp++ -= *sp++;
  } else {
    while ( tp < ftp )
      *tp++ += scalar * *sp++;
  }

  return target;
}

//______________________________________________________________________________
TVectorD &AddElemMult(TVectorD &target,Double_t scalar,
                      const TVectorD &source1,const TVectorD &source2)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) .

  if (!(AreCompatible(target,source1) && AreCompatible(target,source1))) {
    Error("AddElemMult(TVectorD &,Double_t,const TVectorD &,const TVectorD &)",
           "vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp1 = source1.GetMatrixArray();
  const Double_t *       sp2 = source2.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();

  if (scalar == 1.0 ) {
    while ( tp < ftp )
      *tp++ += *sp1++ * *sp2++;
  } else if (scalar == -1.0) {
    while ( tp < ftp )
      *tp++ -= *sp1++ * *sp2++;
  } else {
    while ( tp < ftp )
      *tp++ += scalar * *sp1++ * *sp2++;
  }

  return target;
}

//______________________________________________________________________________
TVectorD &AddElemMult(TVectorD &target,Double_t scalar,
                      const TVectorD &source1,const TVectorD &source2,const TVectorD &select)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) only for those elements
  // where select[i] != 0.0 

  if (!( AreCompatible(target,source1) && AreCompatible(target,source1) &&
         AreCompatible(target,select) )) {
    Error("AddElemMult(TVectorD &,Double_t,const TVectorD &,const TVectorD &,onst TVectorD &)",
           "vector's are incompatible"); 
    target.Invalidate(); 
    return target;
  }

  const Double_t *       sp1 = source1.GetMatrixArray();
  const Double_t *       sp2 = source2.GetMatrixArray();
  const Double_t *       mp  = select.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();

  if (scalar == 1.0 ) {
    while ( tp < ftp ) {
      if (*mp) *tp += *sp1 * *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  } else if (scalar == -1.0) {
    while ( tp < ftp ) {
      if (*mp) *tp -= *sp1 * *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  } else {
    while ( tp < ftp ) {
      if (*mp) *tp += scalar * *sp1 * *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  }

  return target;
}

//______________________________________________________________________________
TVectorD &AddElemDiv(TVectorD &target,Double_t scalar,
                     const TVectorD &source1,const TVectorD &source2)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) .

  if (!(AreCompatible(target,source1) && AreCompatible(target,source1))) {
    Error("AddElemDiv(TVectorD &,Double_t,const TVectorD &,const TVectorD &)",
           "vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp1 = source1.GetMatrixArray();
  const Double_t *       sp2 = source2.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();

  if (scalar == 1.0 ) {
    while ( tp < ftp ) 
      *tp++ += *sp1++ / *sp2++;
  } else if (scalar == -1.0) {
    while ( tp < ftp ) 
      *tp++ -= *sp1++ / *sp2++;
  } else {
    while ( tp < ftp ) 
      *tp++ += scalar * *sp1++ / *sp2++;
  } 

  return target;
}

//______________________________________________________________________________
TVectorD &AddElemDiv(TVectorD &target,Double_t scalar,
                     const TVectorD &source1,const TVectorD &source2,const TVectorD &select)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) only for those elements
  // where select[i] != 0.0 

  if (!( AreCompatible(target,source1) && AreCompatible(target,source1) &&
         AreCompatible(target,select) )) {
    Error("AddElemDiv(TVectorD &,Double_t,const TVectorD &,const TVectorD &,onst TVectorD &)",
           "vector's are incompatible"); 
    target.Invalidate(); 
    return target;
  }

  const Double_t *       sp1 = source1.GetMatrixArray();
  const Double_t *       sp2 = source2.GetMatrixArray();
  const Double_t *       mp  = select.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();

  if (scalar == 1.0 ) {
    while ( tp < ftp ) {
      if (*mp) *tp += *sp1 / *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  } else if (scalar == -1.0) {
    while ( tp < ftp ) {
      if (*mp) *tp -= *sp1 / *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  } else {
    while ( tp < ftp ) {
      if (*mp) *tp += scalar * *sp1 / *sp2;
      mp++; tp++; sp1++; sp2++;
    }
  }

  return target;
}

//______________________________________________________________________________
TVectorD &ElementMult(TVectorD &target,const TVectorD &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    Error("ElementMult(TVectorD &,const TVectorD &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp  = source.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp )
    *tp++ *= *sp++;

  return target;
}

//______________________________________________________________________________
TVectorD &ElementMult(TVectorD &target,const TVectorD &source,const TVectorD &select)
{
  // Multiply target by the source, element-by-element only where select[i] != 0.0 

  if (!(AreCompatible(target,source) && AreCompatible(target,select))) {
    Error("ElementMult(TVectorD &,const TVectorD &,const TVectorD &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp  = source.GetMatrixArray();
  const Double_t *       mp  = select.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp ) {
    if (*mp) *tp *= *sp;
    mp++; tp++; sp++;
  }

  return target;
}

//______________________________________________________________________________
TVectorD &ElementDiv(TVectorD &target,const TVectorD &source)
{
  // Divide target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    Error("ElementDiv(TVectorD &,const TVectorD &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp  = source.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp )
    *tp++ /= *sp++;

  return target;
}

//______________________________________________________________________________
TVectorD &ElementDiv(TVectorD &target,const TVectorD &source,const TVectorD &select)
{ 
  // Divide target by the source, element-by-element only where select[i] != 0.0 
    
  if (!AreCompatible(target,source)) {
    Error("ElementDiv(TVectorD &,const TVectorD &,const TVectorD &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Double_t *       sp  = source.GetMatrixArray();
  const Double_t *       mp  = select.GetMatrixArray();
        Double_t *       tp  = target.GetMatrixArray();
  const Double_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp ) {
    if (*mp) *tp /= *sp;
    mp++; tp++; sp++;
  }

  return target;
}

//______________________________________________________________________________
Bool_t AreCompatible(const TVectorD &v1,const TVectorD &v2,Int_t verbose)
{
  if (!v1.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "vector 1 not initialized");
    return kFALSE;
  }
  if (!v2.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "vector 2 not initialized");
    return kFALSE;
  }

  if (v1.GetNrows() != v2.GetNrows() || v1.GetLwb() != v2.GetLwb()) {
    if (verbose)
      ::Error("AreCompatible", "vectors 1 and 2 not compatible");
    return kFALSE;
  }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t AreCompatible(const TVectorD &v1,const TVectorF &v2,Int_t verbose)
{
  if (!v1.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "vector 1 not initialized");
    return kFALSE;
  }
  if (!v2.IsValid()) {
    if (verbose)
      ::Error("AreCompatible", "vector 2 not initialized");
    return kFALSE;
  }

  if (v1.GetNrows() != v2.GetNrows() || v1.GetLwb() != v2.GetLwb()) {
    if (verbose)
      ::Error("AreCompatible", "vectors 1 and 2 not compatible");
    return kFALSE;
  }

   return kTRUE;
}

//______________________________________________________________________________
void Compare(const TVectorD &v1,const TVectorD &v2)
{
   // Compare two vectors and print out the result of the comparison.

  if (!AreCompatible(v1,v2)) {
    Error("Compare(const TVectorD &,const TVectorD &)","vectors are incompatible");
    return;
  }

  printf("\n\nComparison of two TVectorDs:\n");

  Double_t norm1  = 0;       // Norm of the Matrices
  Double_t norm2  = 0;       // Norm of the Matrices
  Double_t ndiff  = 0;       // Norm of the difference
  Int_t    imax   = 0;       // For the elements that differ most
  Double_t difmax = -1;
  const Double_t *mp1 = v1.GetMatrixArray();    // Vector element pointers
  const Double_t *mp2 = v2.GetMatrixArray();

  for (Int_t i = 0; i < v1.GetNrows(); i++) {
    const Double_t mv1  = *mp1++;
    const Double_t mv2  = *mp2++;
    const Double_t diff = TMath::Abs(mv1-mv2);

    if (diff > difmax) {
      difmax = diff;
      imax = i;
    }
    norm1 += TMath::Abs(mv1);
    norm2 += TMath::Abs(mv2);
    ndiff += TMath::Abs(diff);
  }

  imax += v1.GetLwb();
  printf("\nMaximal discrepancy    \t\t%g",difmax);
  printf("\n   occured at the point\t\t(%d)",imax);
  const Double_t mv1 = v1(imax);
  const Double_t mv2 = v2(imax);
  printf("\n Vector 1 element is    \t\t%g",mv1);
  printf("\n Vector 2 element is    \t\t%g",mv2);
  printf("\n Absolute error v2[i]-v1[i]\t\t%g",mv2-mv1);
  printf("\n Relative error\t\t\t\t%g\n",
         (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Double_t)1e-7));

  printf("\n||Vector 1||   \t\t\t%g",norm1);
  printf("\n||Vector 2||   \t\t\t%g",norm2);
  printf("\n||Vector1-Vector2||\t\t\t\t%g",ndiff);
  printf("\n||Vector1-Vector2||/sqrt(||Vector1|| ||Vector2||)\t%g\n\n",
         ndiff/TMath::Max(TMath::Sqrt(norm1*norm2),1e-7));
}

//______________________________________________________________________________
Bool_t VerifyVectorValue(const TVectorD &v,Double_t val,
                         Int_t verbose,Double_t maxDevAllow)
{
  // Validate that all elements of vector have value val within maxDevAllow .

  Int_t    imax      = 0;
  Double_t maxDevObs = 0;

  for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++) {
    const Double_t dev = TMath::Abs(v(i)-val);
    if (dev > maxDevObs) {
      imax      = i;
      maxDevObs = dev;
    }
  }

  if (maxDevObs == 0)
    return kTRUE;

  if (verbose) {
    printf("Largest dev for (%d); dev = |%g - %g| = %g\n",imax,v(imax),val,maxDevObs);
    if(maxDevObs > maxDevAllow)
      Error("VerifyVectorValue","Deviation > %g\n",maxDevAllow);
  }

  if(maxDevObs > maxDevAllow)
    return kFALSE;
  return kTRUE;
}

//______________________________________________________________________________
Bool_t VerifyVectorIdentity(const TVectorD &v1,const TVectorD &v2,
                            Int_t verbose, Double_t maxDevAllow)
{
  // Verify that elements of the two vectors are equal within maxDevAllow .

  Int_t    imax      = 0;
  Double_t maxDevObs = 0;

  if (!AreCompatible(v1,v2))
    return kFALSE;

  for (Int_t i = v1.GetLwb(); i <= v1.GetUpb(); i++) {
    const Double_t dev = TMath::Abs(v1(i)-v2(i));
    if (dev > maxDevObs) {
      imax      = i;
      maxDevObs = dev;
    }
  }

  if (maxDevObs == 0)
    return kTRUE;

  if (verbose) {
    printf("Largest dev for (%d); dev = |%g - %g| = %g\n",imax,v1(imax),v2(imax),maxDevObs);
    if(maxDevObs > maxDevAllow)
      Error("VerifyVectorIdentity","Deviation > %g\n",maxDevAllow);
  }

  if(maxDevObs > maxDevAllow) {
    return kFALSE;
  }
  return kTRUE;
}

//______________________________________________________________________________
void TVectorD::Streamer(TBuffer &R__b)
{
  // Stream an object of class TVectorD.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    if (R__v > 1) {
      Clear();
      TVectorD::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      if (R__v < 2) MakeValid();
      return;
    }
    //====process old versions before automatic schema evolution
    TObject::Streamer(R__b);
    R__b >> fRowLwb;
    fNrows = R__b.ReadArray(fElements);
    MakeValid();
    R__b.CheckByteCount(R__s, R__c, TVectorD::IsA());
  } else {
    TVectorD::Class()->WriteBuffer(R__b,this);
  }
}
