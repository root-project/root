// @(#)root/matrix:$Name:  $:$Id: TDecompChol.cxx,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Cholesky Decomposition class                                          //
//                                                                       //
// Decompose a symmetric, positive definite matrix A = U^T * U           //
//                                                                       //
// where U is a upper triangular matrix                                  //
//                                                                       //
// The decomposition fails if a diagonal element of fU is <= 0, the      //
// matrix is not positive negative . The matrix fU is made invalid       // 
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "TDecompChol.h"

ClassImp(TDecompChol)

//______________________________________________________________________________
TDecompChol::TDecompChol(const TMatrixDSym &a,Double_t tol)
{
  Assert(a.IsValid());

  fCondition = a.Norm1();
  fTol = a.GetTol();
  if (tol > 0)
    fTol = tol;

  Decompose(a);
}

//______________________________________________________________________________
TDecompChol::TDecompChol(const TMatrixD &a,Double_t tol)
{
  Assert(a.IsValid());

  if (a.GetNrows() != a.GetNcols() || a.GetRowLwb() != a.GetColLwb()) {
    Error("TDecompChol(const TMatrixD &","matrix should be square");
    return;
  }

  fCondition = a.Norm1();
  fTol = a.GetTol();
  if (tol > 0)
    fTol = tol;

  Decompose(a);
}

//______________________________________________________________________________
TDecompChol::TDecompChol(const TDecompChol &another) : TDecompBase(another)
{
  *this = another;
}

//______________________________________________________________________________
Int_t TDecompChol::Decompose(const TMatrixDBase &a)
{
  fU.ResizeTo(a);
  const Int_t     n  = a.GetNrows();
  const Double_t *pA = a.GetMatrixArray();
        Double_t *pU = fU.GetMatrixArray();
  for (Int_t irow = 0; irow < n; irow++)
  {
    const Int_t rowOff = irow*n;
    for (Int_t icol = irow; icol < n; icol++)
    {    
      Double_t sum = 0.;
      for (Int_t l = 0; l < irow; l++)
      { 
        const Int_t off_l = l*n;
        sum += pU[off_l+irow]*pU[off_l+icol];
      }

      pU[rowOff+icol] = pA[rowOff+icol]-sum;
      if (irow == icol)
      {
        if (pU[rowOff+irow] <= 0) {
          fU.Invalidate();
          return kFALSE;
        }
        pU[rowOff+irow] = TMath::Sqrt(pU[rowOff+irow]);
      }
      else
        pU[rowOff+icol] /= pU[rowOff+irow];
    }
  }

  fStatus |= kDecomposed;

  return kTRUE;
}

//______________________________________________________________________________
const TMatrixD TDecompChol::GetMatrix() const
{
  const TMatrixD ut(TMatrixDBase::kTransposed,fU);
  return ut * fU;
}

//______________________________________________________________________________
Bool_t TDecompChol::Solve(TVectorD &b)
{
// Solve equations Ax=b assuming A has been factored by Cholesky. The factor U is
// assumed to be in upper triang of fU. The real fTol is used to determine if
// diagonal element is zero. The solution is returned in b.

  Assert(b.IsValid());
  Assert(fU.IsValid());

  if (fU.GetNrows() != b.GetNrows() || fU.GetRowLwb() != b.GetLwb())
  {
    Error("Solve(TVectorD &","vector and matrix incompatible");
    b.Invalidate();
    return kFALSE;
  }

  const Int_t n = fU.GetNrows();

  const Double_t *pA = fU.GetMatrixArray();
        Double_t *pb = b.GetMatrixArray();

  Int_t i;
  // step 1: Forward substitution
  for (i = n-1; i >= 0; i--) {
    const Int_t off_i = i*n;
    if (pA[off_i+i] < fTol)
    {
      Error("Solve(TVectorD &b)","u[%d,%d]=%.4e < %.4e",i,i,pA[off_i+i],fTol);
      return kFALSE;
    }
    Double_t r = pb[i];
    for (Int_t j = n-1; j >= i; j--) {
      const Int_t off_j = j*n;
      r -= pA[off_j+i]*pb[j];
    }
    pb[i] = r/pA[off_i+i];
  }

 // step 2: Backward substitution
  for (i = 0; i < n; i++) {
    const Int_t off_i = i*n;
    Double_t r = pb[i];
    for (Int_t j = 0; j < i-1; j++)
      r -= pA[off_i+j]*pb[j];
    pb[i] = r/pA[off_i+i];
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompChol::Solve(TMatrixDColumn &cb)
{
  const TMatrixDBase *b = cb.GetMatrix();
  Assert(b->IsValid());
  Assert(fU.IsValid());

  if (fU.GetNrows() != b->GetNrows() || fU.GetRowLwb() != b->GetRowLwb())
  {
    Error("Solve(TColumn &","vector and matrix incompatible");
    return kFALSE;
  }

  const Int_t n = fU.GetNrows();

  const Double_t *pA  = fU.GetMatrixArray();
        Double_t *pcb = cb.GetPtr();
  const Int_t     inc = cb.GetInc();

  Int_t i;
  // step 1: Forward substitution
  for (i = n-1; i >= 0; i--) {
    const Int_t off_i  = i*n;
    const Int_t off_i2 = i*inc;
    if (pA[off_i+i] < fTol)
    {
      Error("Solve(TVectorD &b)","u[%d,%d]=%.4e < %.4e",i,i,pA[off_i+i],fTol);
      return kFALSE;
    }
    Double_t r = pcb[off_i2];
    for (Int_t j = n-1; j >= i; j--) {
      const Int_t off_j = j*n;
      r -= pA[off_j+i]*pcb[j*inc];
    }
    pcb[off_i2] = r/pA[off_i+i];
  }
 
  // step 2: Backward substitution
  for (i = 0; i < n; i++) {
    const Int_t off_i  = i*n;
    const Int_t off_i2 = i*inc;
    Double_t r = pcb[off_i2];
    for (Int_t j = 0; j < i-1; j++)
      r -= pA[off_i+j]*pcb[j*inc];
    pcb[off_i2] = r/pA[off_i+i];
  }

  return kTRUE;
}

//______________________________________________________________________________
void TDecompChol::Det(Double_t &d1,Double_t &d2)
{
  // determinant is square of diagProd of cholesky factor

  if ( !( fStatus & kDetermined ) ) {
    if ( fStatus & kSingular ) {
      fDet1 = 0.0;
      fDet2 = 0.0;
    } else 
      TDecompBase::Det(d1,d2);
    // square det as calculated by above
    fDet1 *= fDet1;
    fDet2 += fDet2;
    fStatus |= kDetermined;
  }
  d1 = fDet1;
  d2 = fDet2;
}

//______________________________________________________________________________
TDecompChol &TDecompChol::operator=(const TDecompChol &source)
{ 
  if (this != &source) {
    TDecompBase::operator=(source);
    fU.ResizeTo(source.fU);
    fU = source.fU;
  }
  return *this;
}     

//______________________________________________________________________________
TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b)
{
  // Solve A . x = b for vector x where
  //   A : (m x n) matrix, m >= n
  //   b : (m)     vector
  //   x : (n)     vector

  TDecompChol ch(TMatrixD(A,TMatrixDBase::kTransposeMult,A));
  TVectorD x = TMatrixD(TMatrixDBase::kTransposed,A)*b;
  ch.Solve(x);
  return x;
}

//______________________________________________________________________________
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B)
{
  // Solve A . x = B for matrix x where
  //   A : (m x n ) matrix, m >= n
  //   B : (m x nb) matrix, nb >= 1
  //   x : (n x nb) matrix

  TDecompChol ch(TMatrixD(A,TMatrixDBase::kTransposeMult,A));
  TMatrixD X(A,TMatrixDBase::kTransposeMult,B);
  ch.MultiSolve(X);
  return X;
}
