// @(#)root/matrix:$Name:  $:$Id: TVectorF.cxx,v 1.19 2004/05/27 06:39:53 brun Exp $
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
// TVectorF                                                             //
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

#include "TVectorF.h"

ClassImp(TVectorF)

//______________________________________________________________________________
void TVectorF::Delete_m(Int_t size,Float_t *&m)
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
Float_t* TVectorF::New_m(Int_t size)
{
  if (size == 0) return 0;
  else {
    if ( size <= kSizeMax )
      return fDataStack;
    else {
      Float_t *heap = new Float_t[size];
      return heap;
    }
  }
}

//______________________________________________________________________________
Int_t TVectorF::Memcpy_m(Float_t *newp,const Float_t *oldp,Int_t copySize,
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
      memcpy(newp,oldp,copySize*sizeof(Float_t));
    }
  }
  return 0;
}

//______________________________________________________________________________
void TVectorF::Allocate(Int_t nrows,Int_t row_lwb,Int_t init)
{
  // Allocate new vector. Arguments are number of rows and row
  // lowerbound (0 default).

  Invalidate();
  Assert(nrows > 0);

  SetBit(TMatrixFBase::kStatus);
  fNrows   = nrows;
  fRowLwb  = row_lwb;
  fIsOwner = kTRUE;

  fElements = New_m(fNrows);
  if (init)
    memset(fElements,0,fNrows*sizeof(Float_t));
}

//______________________________________________________________________________
TVectorF::TVectorF(Int_t n)
{
  Allocate(n,0,1);
}

//______________________________________________________________________________
TVectorF::TVectorF(Int_t lwb,Int_t upb)
{
  Allocate(upb-lwb+1,lwb,1);
}

//______________________________________________________________________________
TVectorF::TVectorF(Int_t n,const Float_t *elements)
{
  Allocate(n,0);
  SetElements(elements);
}

//______________________________________________________________________________
TVectorF::TVectorF(Int_t lwb,Int_t upb,const Float_t *elements)
{
  Allocate(upb-lwb+1,lwb);
  SetElements(elements);
}

//______________________________________________________________________________
TVectorF::TVectorF(const TVectorF &another) : TObject(another)
{
  Assert(another.IsValid());
  Allocate(another.GetUpb()-another.GetLwb()+1,another.GetLwb());
  *this = another;
}

//______________________________________________________________________________
TVectorF::TVectorF(const TVectorD &another) : TObject(another)
{
  Assert(another.IsValid());
  Allocate(another.GetUpb()-another.GetLwb()+1,another.GetLwb());
  *this = another;
}

//______________________________________________________________________________
TVectorF::TVectorF(const TMatrixFRow_const &mr) : TObject()
{
  const TMatrixFBase *mt = mr.GetMatrix();
  Assert(mt->IsValid());
  Allocate(mt->GetColUpb()-mt->GetColLwb()+1,mt->GetColLwb());
  *this = mr;
}

//______________________________________________________________________________
TVectorF::TVectorF(const TMatrixFColumn_const &mc) : TObject()
{
  const TMatrixFBase *mt = mc.GetMatrix();
  Assert(mt->IsValid());
  Allocate(mt->GetRowUpb()-mt->GetRowLwb()+1,mt->GetRowLwb());
  *this = mc;
}

//______________________________________________________________________________
TVectorF::TVectorF(const TMatrixFDiag_const &md) : TObject()
{
  const TMatrixFBase *mt = md.GetMatrix();
  Assert(mt->IsValid());
  Allocate(TMath::Min(mt->GetNrows(),mt->GetNcols()));
  *this = md;
}

//______________________________________________________________________________
TVectorF::TVectorF(Int_t lwb,Int_t upb,Float_t va_(iv1), ...)
{
  // Make a vector and assign initial values. Argument list should contain
  // Float_t values to assign to vector elements. The list must be
  // terminated by the string "END". Example:
  // TVectorF foo(1,3,0.0,1.0,1.5,"END");

  const Int_t no_rows = upb-lwb+1;
  Assert(no_rows);
  Allocate(no_rows,lwb);

  va_list args;
  va_start(args,va_(iv1));             // Init 'args' to the beginning of
                                       // the variable length list of args

  (*this)(lwb) = iv1;
  for (Int_t i = lwb+1; i <= upb; i++)
    (*this)(i) = (Float_t)va_arg(args,Double_t);

  if (strcmp((char *)va_arg(args,char *),"END"))
    Error("TVectorF(Int_t, Int_t, ...)", "argument list must be terminated by \"END\"");

  va_end(args);
}

//______________________________________________________________________________
void TVectorF::ResizeTo(Int_t lwb,Int_t upb)
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

    Float_t     *elements_old = GetMatrixArray();
    const Int_t  nrows_old    = fNrows;
    const Int_t  rowLwb_old   = fRowLwb;

    Allocate(new_nrows,lwb);
    Assert(IsValid());
    if (fNrows > kSizeMax || nrows_old > kSizeMax)
      memset(GetMatrixArray(),0,fNrows*sizeof(Float_t));
    else if (fNrows > nrows_old)
      memset(GetMatrixArray()+nrows_old,0,(fNrows-nrows_old)*sizeof(Float_t));

    // Copy overlap
    const Int_t rowLwb_copy = TMath::Max(fRowLwb,rowLwb_old);
    const Int_t rowUpb_copy = TMath::Min(fRowLwb+fNrows-1,rowLwb_old+nrows_old-1);
    const Int_t nrows_copy  = rowUpb_copy-rowLwb_copy+1;

    const Int_t nelems_new = fNrows;
    Float_t *elements_new = GetMatrixArray();
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
void TVectorF::Use(Int_t n,Float_t *data)
{
  Assert(n > 0);

  Clear();
  fNrows    = n;
  fRowLwb   = 0;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
void TVectorF::Use(Int_t lwb,Int_t upb,Float_t *data)
{
  Assert(upb >= lwb);

  Clear();
  fNrows    = upb-lwb+1;
  fRowLwb   = lwb;
  fElements = data;
  fIsOwner  = kFALSE;
}

//______________________________________________________________________________
TVectorF TVectorF::GetSub(Int_t row_lwb,Int_t row_upb,Option_t *option) const
{
  // Get subvector [row_lwb..row_upb]; The indexing range of the
  // returned vector depends on the argument option:
  //
  // option == "S" : return [0..row_upb-row_lwb+1] (default)
  // else          : return [row_lwb..row_upb]

  Assert(IsValid());
  if (row_lwb < fRowLwb || row_lwb > fRowLwb+fNrows-1) {
    Error("GetSub","row_lwb out of bounds");
    return TVectorF();
  }
  if (row_upb < fRowLwb || row_upb > fRowLwb+fNrows-1) {
    Error("GetSub","row_upb out of bounds");
    return TVectorF();
  }
  if (row_upb < row_lwb) {
    Error("GetSub","row_upb < row_lwb");
    return TVectorF();
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

  TVectorF sub(row_lwb_sub,row_upb_sub);
  const Int_t nrows_sub = row_upb_sub-row_lwb_sub+1;

  const Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb);
        Float_t *bp = sub.GetMatrixArray();

  for (Int_t irow = 0; irow < nrows_sub; irow++)
      *bp++ = *ap++;

  return sub;
}

//______________________________________________________________________________
void TVectorF::SetSub(Int_t row_lwb,const TVectorF &source)
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

  const Float_t *bp = source.GetMatrixArray();
        Float_t *ap = this->GetMatrixArray()+(row_lwb-fRowLwb);

  for (Int_t irow = 0; irow < nRows_source; irow++)
    *ap++ = *bp++;
}

//______________________________________________________________________________
TVectorF &TVectorF::Zero()
{
  Assert(IsValid());
  memset(this->GetMatrixArray(),0,fNrows*sizeof(Float_t));
  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::Abs()
{
  // Take an absolute value of a vector, i.e. apply Abs() to each element.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep = TMath::Abs(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::Sqr()
{
  // Square each element of the vector.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep = (*ep) * (*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::Sqrt()
{
   // Take square root of all elements.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    Assert(*ep >= 0);
    *ep = TMath::Sqrt(*ep);
    ep++;
  }

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::Invert()
{ 
   // v[i] = 1/v[i]
    
  Assert(IsValid());
  
        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    Assert(*ep != 0.0);
    *ep = 1./ *ep;
    ep++;
  }
   
  return *this;
} 

//______________________________________________________________________________
TVectorF &TVectorF::SelectNonZeros(const TVectorF &select)
{  
  if (!AreCompatible(*this,select)) {
    Error("SelectNonZeros(const TVectorF &","vector's not compatible");
    Invalidate();
    return *this;
  }
 
  const Float_t *sp = select.GetMatrixArray();
        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    if (*sp == 0.0)
      *ep = 0.0;
    sp++; ep++;
  }

  return *this;
}  

//______________________________________________________________________________
Float_t TVectorF::Norm1() const
{
  // Compute the 1-norm of the vector SUM{ |v[i]| }.

  Assert(IsValid());

  Float_t norm = 0;
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    norm += TMath::Abs(*ep++);

  return norm;
}

//______________________________________________________________________________
Float_t TVectorF::Norm2Sqr() const
{
  // Compute the square of the 2-norm SUM{ v[i]^2 }.

  Assert(IsValid());

  Float_t norm = 0;
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    norm += (*ep) * (*ep);
    ep++;
  }

  return norm;
}

//______________________________________________________________________________
Float_t TVectorF::NormInf() const
{
  // Compute the infinity-norm of the vector MAX{ |v[i]| }.

  Assert(IsValid());

  Float_t norm = 0;
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    norm = TMath::Max(norm,TMath::Abs(*ep++));

  return norm;
}

//______________________________________________________________________________
Int_t TVectorF::NonZeros() const
{
  // Compute the number of elements != 0.0

  Assert(IsValid());

  Int_t nr_nonzeros = 0;
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (*ep++) nr_nonzeros++;

  return nr_nonzeros;
}

//______________________________________________________________________________
Float_t TVectorF::Sum() const
{
  // Compute sum of elements 

  Assert(IsValid());

  Float_t sum = 0.0;
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    sum += *ep++;

  return sum;
}

//______________________________________________________________________________
Float_t TVectorF::Min() const
{
  // return minimum vector element value

  Assert(IsValid());

  const Int_t index = TMath::LocMin(fNrows,fElements);
  return fElements[index];
}

//______________________________________________________________________________
Float_t TVectorF::Max() const
{
  // return maximum vector element value

  Assert(IsValid());

  const Int_t index = TMath::LocMax(fNrows,fElements);
  return fElements[index];
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(const TVectorF &source)
{
  // Notice that this assignment does NOT change the ownership :
  // if the storage space was adopted, source is copied to
  // this space .

  if (!AreCompatible(*this,source)) {
    Error("operator=(const TVectorF &)","vectors not compatible");
    Invalidate();
    return *this;
  }

  if (this != &source) {
    TObject::operator=(source);
    memcpy(fElements,source.GetMatrixArray(),fNrows*sizeof(Float_t));
  }
  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(const TVectorD &source)
{
  if (!AreCompatible(*this,source)) {
    Error("operator=(const TVectorD &)","vectors not compatible");
    Invalidate();
    return *this;
  }

  if (dynamic_cast<TVectorD *>(this) != &source) {
    TObject::operator=(source);
    const Double_t * const ps = source.GetMatrixArray();
          Float_t  * const pt = GetMatrixArray();
    for (Int_t i = 0; i < fNrows; i++)
      pt[i] = (Float_t) ps[i];
  }
  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(const TMatrixFRow_const &mr)
{
  // Assign a matrix row to a vector.

  Assert(IsValid());
  const TMatrixFBase *mt = mr.GetMatrix();
  Assert(mt->IsValid());

  if (mt->GetColLwb() != fRowLwb || mt->GetNcols() != fNrows) {
    Error("operator=(const TMatrixFRow_const &)","vector and row not compatible");
    Invalidate();
    return *this;
  }

  const Int_t inc   = mr.GetInc();
  const Float_t *rp = mr.GetPtr();              // Row ptr
        Float_t *ep = this->GetMatrixArray();   // Vector ptr
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *rp;
     rp += inc;
  }

  Assert(rp == mr.GetPtr()+mt->GetNcols());

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(const TMatrixFColumn_const &mc)
{
   // Assign a matrix column to a vector.

  Assert(IsValid());
  const TMatrixFBase *mt = mc.GetMatrix();
  Assert(mt->IsValid());

  if (mt->GetRowLwb() != fRowLwb || mt->GetNrows() != fNrows) {
    Error("operator=(const TMatrixFColumn_const &)","vector and column not compatible");
    Invalidate();
    return *this;
  }

  const Int_t inc   = mc.GetInc();
  const Float_t *cp = mc.GetPtr();              // Column ptr
        Float_t *ep = this->GetMatrixArray();   // Vector ptr
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *cp;
     cp += inc;
  }

  Assert(cp == mc.GetPtr()+mt->GetNoElements());

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(const TMatrixFDiag_const &md)
{
  // Assign the matrix diagonal to a vector.

  Assert(IsValid());
  const TMatrixFBase *mt = md.GetMatrix();
  Assert(mt->IsValid());

  if (md.GetNdiags() != fNrows) {
    Error("operator=(const TMatrixFDiag_const &)","vector and matrix-diagonal not compatible");
    Invalidate();
    return *this;
  }

  const Int_t    inc = md.GetInc();
  const Float_t *dp  = md.GetPtr();              // Diag ptr
        Float_t *ep  = this->GetMatrixArray();   // Vector ptr
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    *ep++ = *dp;
     dp += inc;
  }

  Assert(dp < md.GetPtr()+mt->GetNoElements()+inc);

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator=(Float_t val)
{
  // Assign val to every element of the vector.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ = val;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator+=(Float_t val)
{
  // Add val to every element of the vector.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ += val;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator-=(Float_t val)
{
  // Subtract val from every element of the vector.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ -= val;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator*=(Float_t val)
{
  // Multiply every element of the vector with val.

  Assert(IsValid());

        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ *= val;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator+=(const TVectorF &source)
{
  // Add vector source

  if (!AreCompatible(*this,source)) {
    Error("operator+=(const TVectorF &)","vector's not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp = source.GetMatrixArray();
        Float_t *tp = this->GetMatrixArray();
  const Float_t * const tp_last = tp+fNrows;
  while (tp < tp_last)
    *tp++ += *sp++;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator-=(const TVectorF &source)
{
  // Subtract vector source

  if (!AreCompatible(*this,source)) {
    Error("operator-=(const TVectorF &)","vector's not compatible");
    Invalidate();
    return *this;
  }

  const Float_t *sp = source.GetMatrixArray();
        Float_t *tp = this->GetMatrixArray();
  const Float_t * const tp_last = tp+fNrows;
  while (tp < tp_last)
    *tp++ -= *sp++;

  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::operator*=(const TMatrixF &a)
{
  // "Inplace" multiplication target = A*target. A needn't be a square one
  // If target has to be resized, it should own the storage: fIsOwner = kTRUE

  Assert(IsValid());
  Assert(a.IsValid());

  if (a.GetNcols() != fNrows || a.GetColLwb() != fRowLwb) {
    Error("operator*=(const TMatrixF &)","vector and matrix incompatible");
    Invalidate();
    return *this;
  }

  if ((fNrows != a.GetNrows() || fRowLwb != a.GetRowLwb()) && !fIsOwner) {
    Error("operator*=(const TMatrixF &)","vector has to be resized but not owner");
    Invalidate();
    return *this;
  }

  const Int_t nrows_old = fNrows;
  Float_t *elements_old;
  if (nrows_old <= kSizeMax) {
    elements_old = new Float_t[nrows_old];
    memcpy(elements_old,fElements,nrows_old*sizeof(Float_t));
  }
  else
    elements_old = fElements;

  fRowLwb = a.GetRowLwb();
  Assert((fNrows = a.GetNrows()) > 0);

  Allocate(fNrows,fRowLwb);

  const Float_t *mp = a.GetMatrixArray();     // Matrix row ptr
        Float_t *tp = this->GetMatrixArray(); // Target vector ptr
#ifdef CBLAS
  cblas_sgemv(CblasRowMajor,CblasNoTrans,a.GetNrows(),a.GetNcols(),1.0,mp,
              a.GetNcols(),elements_old,1,0.0,tp,1);
#else
  const Float_t * const tp_last = tp+fNrows;
  while (tp < tp_last) {
    Float_t sum = 0;
    for (const Float_t *sp = elements_old; sp < elements_old+nrows_old; )
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
TVectorF &TVectorF::operator*=(const TMatrixFSym &a)
{
  // "Inplace" multiplication target = A*target. A is symmetric .
  // vector size will not change

  Assert(IsValid());
  Assert(a.IsValid());

  if (a.GetNcols() != fNrows || a.GetColLwb() != fRowLwb) {
    Error("operator*=(const TMatrixFSym &)","vector and matrix incompatible");
    Invalidate();
    return *this;
  }

  Float_t * const elements_old = new Float_t[fNrows];
  memcpy(elements_old,fElements,fNrows*sizeof(Float_t));
  memset(fElements,0,fNrows*sizeof(Float_t));

  const Float_t *mp1 = a.GetMatrixArray(); // Matrix row ptr
        Float_t *tp1 = fElements;       // Target vector ptr
#ifdef CBLAS
  cblas_ssymv(CblasRowMajor,CblasUpper,fNrows,1.0,mp1,
              fNrows,elements_old,1,0.0,tp1,1);
#else
  const Float_t *mp2;
  const Float_t *sp1 = elements_old;
  const Float_t *sp2 = sp1;
        Float_t *tp2 = tp1;       // Target vector ptr

  for (Int_t i = 0; i < fNrows; i++) {
    Float_t vec_i = *sp1++;
    *tp1 += *mp1 * vec_i;
    Float_t tmp = 0.0;
    mp2 = mp1+1;
    sp2 = sp1;
    tp2 = tp1+1;
    for (Int_t j = i+1; j < fNrows; j++) {
      const Float_t a_ij = *mp2++;
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
Bool_t TVectorF::operator==(Float_t val) const
{
  // Are all vector elements equal to val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ == val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::operator!=(Float_t val) const
{
  // Are all vector elements not equal to val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ != val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::operator<(Float_t val) const
{
  // Are all vector elements < val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ < val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::operator<=(Float_t val) const
{
  // Are all vector elements <= val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ <= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::operator>(Float_t val) const
{
  // Are all vector elements > val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ > val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::operator>=(Float_t val) const
{
  // Are all vector elements >= val?

  Assert(IsValid());

  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (!(*ep++ >= val))
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TVectorF::MatchesNonZeroPattern(const TVectorF &select)
{
  if (!AreCompatible(*this,select)) {
    Error("MatchesNonZeroPattern(const TVectorF&)","vector's not compatible");
    return kFALSE;
  }

  const Float_t *sp = select.GetMatrixArray();
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) 
    if (*sp++ == 0.0 && *ep++ != 0.0)
      return kFALSE;
  
  return kTRUE;
} 

//______________________________________________________________________________
Bool_t TVectorF::SomePositive(const TVectorF &select)
{
  if (!AreCompatible(*this,select)) {
    Error("SomePositive(const TVectorF&)","vector's not compatible");
    return kFALSE;
  }

  const Float_t *sp = select.GetMatrixArray();
  const Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    if (*sp++ != 0.0 && *ep++ <= 0.0)
      return kFALSE;

  return kTRUE;
}

//______________________________________________________________________________
void TVectorF::AddSomeConstant(Float_t val,const TVectorF &select)
{
  if (!AreCompatible(*this,select))
    Error("AddSomeConstant(Float_t,const TVectorF &)","vector's not compatible");

  const Float_t *sp = select.GetMatrixArray();
        Float_t *ep = this->GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp) {
    if (*sp)
      *ep += val;
    sp++; ep++;
  }
}

//______________________________________________________________________________
void TVectorF::Randomize(Float_t alpha,Float_t beta,Double_t &seed)
{
  // randomize vector elements value
  
  Assert(IsValid());
  
  const Float_t scale = beta-alpha;
  const Float_t shift = alpha/scale;

        Float_t *       ep = GetMatrixArray();
  const Float_t * const fp = ep+fNrows;
  while (ep < fp)
    *ep++ = scale*(Frand(seed)+shift);
}

//______________________________________________________________________________
TVectorF &TVectorF::Apply(const TElementActionF &action)
{
  // Apply action to each element of the vector.

  Assert(IsValid());
  for (Float_t *ep = fElements; ep < fElements+fNrows; ep++)
    action.Operation(*ep);
  return *this;
}

//______________________________________________________________________________
TVectorF &TVectorF::Apply(const TElementPosActionF &action)
{
  // Apply action to each element of the vector. In action the location
  // of the current element is known.

  Assert(IsValid());

  Float_t *ep = fElements;
  for (action.fI = fRowLwb; action.fI < fRowLwb+fNrows; action.fI++)
    action.Operation(*ep++);

  Assert(ep == fElements+fNrows);

  return *this;
}

//______________________________________________________________________________
void TVectorF::Draw(Option_t *option)
{
  // Draw this vector using an intermediate histogram
  // The histogram is named "TVectorF" by default and no title

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
void TVectorF::Print(Option_t *flag) const
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
Bool_t operator==(const TVectorF &v1,const TVectorF &v2)
{
  // Check to see if two vectors are identical.

  if (!AreCompatible(v1,v2)) return kFALSE;
  return (memcmp(v1.GetMatrixArray(),v2.GetMatrixArray(),v1.GetNrows()*sizeof(Float_t)) == 0);
}

//______________________________________________________________________________
Float_t operator*(const TVectorF &v1,const TVectorF &v2)
{
  // Compute the scalar product.

  if (!AreCompatible(v1,v2)) {
    Error("operator*(const TVectorF &,const TVectorF &)","vector's are incompatible");
    return 0.0;
  }

  const Float_t *v1p = v1.GetMatrixArray();
  const Float_t *v2p = v2.GetMatrixArray();

  Float_t sum = 0.0;
  const Float_t * const fv1p = v1p+v1.GetNrows();
  while (v1p < fv1p)
    sum += *v1p++ * *v2p++;

  return sum;
}

//______________________________________________________________________________
TVectorF operator+(const TVectorF &source1,const TVectorF &source2)
{
  TVectorF target = source1;
  target += source2;
  return target;
}

//______________________________________________________________________________
TVectorF operator-(const TVectorF &source1,const TVectorF &source2)
{
  TVectorF target = source1;
  target -= source2;
  return target;
}

//______________________________________________________________________________
TVectorF operator*(const TMatrixF &a,const TVectorF &source)
{
  TVectorF target = source;
  target *= a;
  return target;
}

//______________________________________________________________________________
TVectorF operator*(const TMatrixFSym &a,const TVectorF &source)
{
  TVectorF target = source;
  target *= a;
  return target;
}

//______________________________________________________________________________
TVectorF operator*(Float_t val,const TVectorF &source)
{
  TVectorF target = source;
  target *= val;
  return target;
}

//______________________________________________________________________________
TVectorF &Add(TVectorF &target,Float_t scalar,const TVectorF &source)
{
  // Modify addition: target += scalar * source.

  if (!AreCompatible(target,source)) {
    Error("Add(TVectorF &,Float_t,const TVectorF &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp  = source.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();
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
TVectorF &AddElemMult(TVectorF &target,Float_t scalar,
                      const TVectorF &source1,const TVectorF &source2)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) .

  if (!(AreCompatible(target,source1) && AreCompatible(target,source1))) {
    Error("AddElemMult(TVectorF &,Float_t,const TVectorF &,const TVectorF &)",
           "vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp1 = source1.GetMatrixArray();
  const Float_t *       sp2 = source2.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();

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
TVectorF &AddElemMult(TVectorF &target,Float_t scalar,
                      const TVectorF &source1,const TVectorF &source2,const TVectorF &select)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) only for those elements
  // where select[i] != 0.0 

  if (!( AreCompatible(target,source1) && AreCompatible(target,source1) &&
         AreCompatible(target,select) )) {
    Error("AddElemMult(TVectorF &,Float_t,const TVectorF &,const TVectorF &,const TVectorF &)",
           "vector's are incompatible"); 
    target.Invalidate(); 
    return target;
  }

  const Float_t *       sp1 = source1.GetMatrixArray();
  const Float_t *       sp2 = source2.GetMatrixArray();
  const Float_t *       mp  = select.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();

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
TVectorF &AddElemDiv(TVectorF &target,Float_t scalar,
                     const TVectorF &source1,const TVectorF &source2)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) .

  if (!(AreCompatible(target,source1) && AreCompatible(target,source1))) {
    Error("AddElemMult(TVectorF &,Float_t,const TVectorF &,const TVectorF &)",
           "vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp1 = source1.GetMatrixArray();
  const Float_t *       sp2 = source2.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();

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
TVectorF &AddElemDiv(TVectorF &target,Float_t scalar,
                     const TVectorF &source1,const TVectorF &source2,const TVectorF &select)
{
  // Modify addition: target += scalar * ElementMult(source1,source2) only for those elements
  // where select[i] != 0.0 

  if (!( AreCompatible(target,source1) && AreCompatible(target,source1) &&
         AreCompatible(target,select) )) {
    Error("AddElemDiv(TVectorF &,Float_t,const TVectorF &,const TVectorF &,const TVectorF &)",
           "vector's are incompatible"); 
    target.Invalidate(); 
    return target;
  }

  const Float_t *       sp1 = source1.GetMatrixArray();
  const Float_t *       sp2 = source2.GetMatrixArray();
  const Float_t *       mp  = select.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();

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
TVectorF &ElementMult(TVectorF &target,const TVectorF &source)
{
  // Multiply target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    Error("ElementMult(TVectorF &,const TVectorF &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp  = source.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp )
    *tp++ *= *sp++;

  return target;
}

//______________________________________________________________________________
TVectorF &ElementMult(TVectorF &target,const TVectorF &source,const TVectorF &select)
{
  // Multiply target by the source, element-by-element only where select[i] != 0.0

  if (!(AreCompatible(target,source) && AreCompatible(target,select))) {
    Error("ElementMult(TVectorF &,const TVectorF &,const TVectorF &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp  = source.GetMatrixArray();
  const Float_t *       mp  = select.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp ) {
    if (*mp) *tp *= *sp;
    mp++; tp++; sp++;
  }

  return target;
}

//______________________________________________________________________________
TVectorF &ElementDiv(TVectorF &target,const TVectorF &source)
{
  // Divide target by the source, element-by-element.

  if (!AreCompatible(target,source)) {
    Error("ElementDiv(TVectorF &,const TVectorF &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp  = source.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();
 while ( tp < ftp )
    *tp++ /= *sp++;

  return target;
}

//______________________________________________________________________________
TVectorF &ElementDiv(TVectorF &target,const TVectorF &source,const TVectorF &select)
{
  // Divide target by the source, element-by-element only where select[i] != 0.0

  if (!AreCompatible(target,source)) {
    Error("ElementDiv(TVectorF &,const TVectorF &,const TVectorF &)","vector's are incompatible");
    target.Invalidate();
    return target;
  }

  const Float_t *       sp  = source.GetMatrixArray();
  const Float_t *       mp  = select.GetMatrixArray();
        Float_t *       tp  = target.GetMatrixArray();
  const Float_t * const ftp = tp+target.GetNrows();
  while ( tp < ftp ) {
    if (*mp) *tp /= *sp;
    mp++; tp++; sp++;
  }

  return target;
}

//______________________________________________________________________________
Bool_t AreCompatible(const TVectorF &v1,const TVectorF &v2,Int_t verbose)
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
//
//______________________________________________________________________________
Bool_t AreCompatible(const TVectorF &v1,const TVectorD &v2,Int_t verbose)
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
void Compare(const TVectorF &v1,const TVectorF &v2)
{
   // Compare two vectors and print out the result of the comparison.

  if (!AreCompatible(v1,v2)) {
    Error("Compare(const TVectorF &,const TVectorF &)","vectors are incompatible");
    return;
  }

  printf("\n\nComparison of two TVectorFs:\n");

  Float_t norm1  = 0;       // Norm of the Matrices
  Float_t norm2  = 0;       // Norm of the Matrices
  Float_t ndiff  = 0;       // Norm of the difference
  Int_t    imax  = 0;       // For the elements that differ most
  Float_t difmax = -1;
  const Float_t *mp1 = v1.GetMatrixArray();    // Vector element pointers
  const Float_t *mp2 = v2.GetMatrixArray();

  for (Int_t i = 0; i < v1.GetNrows(); i++) {
    const Float_t mv1  = *mp1++;
    const Float_t mv2  = *mp2++;
    const Float_t diff = TMath::Abs(mv1-mv2);

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
  const Float_t mv1 = v1(imax);
  const Float_t mv2 = v2(imax);
  printf("\n Vector 1 element is    \t\t%g",mv1);
  printf("\n Vector 2 element is    \t\t%g",mv2);
  printf("\n Absolute error v2[i]-v1[i]\t\t%g",mv2-mv1);
  printf("\n Relative error\t\t\t\t%g\n",
         (mv2-mv1)/TMath::Max(TMath::Abs(mv2+mv1)/2,(Float_t)1e-7));

  printf("\n||Vector 1||   \t\t\t%g",norm1);
  printf("\n||Vector 2||   \t\t\t%g",norm2);
  printf("\n||Vector1-Vector2||\t\t\t\t%g",ndiff);
  printf("\n||Vector1-Vector2||/sqrt(||Vector1|| ||Vector2||)\t%g\n\n",
         ndiff/TMath::Max(TMath::Sqrt(norm1*norm2),1e-7));
}

//______________________________________________________________________________
Bool_t VerifyVectorValue(const TVectorF &v,Float_t val,
                         Int_t verbose,Float_t maxDevAllow)
{
  // Validate that all elements of vector have value val within maxDevAllow .

  Int_t   imax      = 0;
  Float_t maxDevObs = 0;

  for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++) {
    const Float_t dev = TMath::Abs(v(i)-val);
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
Bool_t VerifyVectorIdentity(const TVectorF &v1,const TVectorF &v2,
                            Int_t verbose, Float_t maxDevAllow)
{
  // Verify that elements of the two vectors are equal within maxDevAllow .

  Int_t   imax      = 0;
  Float_t maxDevObs = 0;

  if (!AreCompatible(v1,v2))
    return kFALSE;

  for (Int_t i = v1.GetLwb(); i <= v1.GetUpb(); i++) {
    const Float_t dev = TMath::Abs(v1(i)-v2(i));
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
void TVectorF::Streamer(TBuffer &R__b)
{
  // Stream an object of class TVectorF.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    if (R__v > 1) {
      Clear();
      TVectorF::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      MakeValid();
      return;
    }
    //====process old versions before automatic schema evolution
    TObject::Streamer(R__b);
    R__b >> fRowLwb;
    fNrows = R__b.ReadArray(fElements);
    MakeValid();
    R__b.CheckByteCount(R__s, R__c, TVectorF::IsA());
  } else {
    TVectorF::Class()->WriteBuffer(R__b,this);
  }
}

//______________________________________________________________________________
void TVector::Streamer(TBuffer &R__b)
{
  // Stream an object of class TVector.

  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    Version_t R__v = R__b.ReadVersion(&R__s,&R__c);
    if (R__v > 2) {
      Clear();
      TVectorF::Class()->ReadBuffer(R__b,this,R__v,R__s,R__c);
      return;
    }
    //====process old version 2
    if (R__v > 1) {
      Clear();
      TObject::Streamer(R__b);
      R__b >> fNrows;
      R__b >> fRowLwb;
      Char_t isArray;
      R__b >> isArray;
      if (fNrows) {
        fElements = new Float_t[fNrows];
        R__b.ReadFastArray(fElements,fNrows);
      }
      R__b.CheckByteCount(R__s, R__c, TVector::IsA());
      MakeValid();
      return;
    }
    //====process old version 1
    TObject::Streamer(R__b);
    R__b >> fRowLwb;
    fNrows = R__b.ReadArray(fElements);
    MakeValid();
    R__b.CheckByteCount(R__s, R__c, TVector::IsA());
  } else {
    TVectorF::Class()->WriteBuffer(R__b,this);
  }
}
