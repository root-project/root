// @(#)root/matrix:$Name:  $:$Id: TDecompChol.cxx,v 1.4 2004/02/04 17:12:44 brun Exp $
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

  fRowLwb = a.GetRowLwb();
  fColLwb = a.GetColLwb();
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

  fRowLwb = a.GetRowLwb();
  fColLwb = a.GetColLwb();
  Decompose(a);
}

//______________________________________________________________________________
TDecompChol::TDecompChol(const TDecompChol &another) : TDecompBase(another)
{
  *this = another;
}

//______________________________________________________________________________
Int_t TDecompChol::Decompose(const TMatrixD &a)
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
          Error("Decompose(const TMatrixDBase &","matrix not positive definite");
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
  if (fStatus & kSingular)
    return TMatrixD();
  if ( !( fStatus & kDecomposed ) )
      return TMatrixD();

  const TMatrixD ut(TMatrixDBase::kTransposed,fU);
  return ut * fU;
}

//______________________________________________________________________________
Bool_t TDecompChol::Solve(TVectorD &b)
{
// Solve equations Ax=b assuming A has been factored by Cholesky. The factor U is
// assumed to be in upper triang of fU. fTol is used to determine if diagonal
// element is zero. The solution is returned in b.

  Assert(b.IsValid());
  Assert(fStatus & kDecomposed);

  if (fU.GetNrows() != b.GetNrows() || fU.GetRowLwb() != b.GetLwb())
  {
    Error("Solve(TVectorD &","vector and matrix incompatible");
    b.Invalidate();
    return kFALSE;
  }

  const Int_t n = fU.GetNrows();

  const Double_t *pU = fU.GetMatrixArray();
        Double_t *pb = b.GetMatrixArray();

  Int_t i;
  // step 1: Forward substitution on U^T
  for (i = 0; i < n; i++) {
    const Int_t off_i = i*n;
    if (pU[off_i+i] < fTol)
    {
      Error("Solve(TVectorD &b)","u[%d,%d]=%.4e < %.4e",i,i,pU[off_i+i],fTol);
      return kFALSE;
    }
    Double_t r = pb[i];
    for (Int_t j = 0; j < i; j++) {
      const Int_t off_j = j*n;
      r -= pU[off_j+i]*pb[j];
    }
    pb[i] = r/pU[off_i+i];
  }

 // step 2: Backward substitution on U
  for (i = n-1; i >= 0; i--) {
    const Int_t off_i = i*n;
    Double_t r = pb[i];
    for (Int_t j = i+1; j < n; j++)
      r -= pU[off_i+j]*pb[j];
    pb[i] = r/pU[off_i+i];
  }

  return kTRUE;
}

//______________________________________________________________________________
TVectorD TDecompChol::Solve(const TVectorD &b,Bool_t &ok)
{    
// Solve equations Ax=b assuming A has been factored by Cholesky. The factor U is
// assumed to be in upper triang of fU. fTol is used to determine if diagonal
// element is zero.
    
  TVectorD x = b; 
  ok = Solve(x);
      
  return x;
}

//______________________________________________________________________________
Bool_t TDecompChol::Solve(TMatrixDColumn &cb)
{ 
  const TMatrixDBase *b = cb.GetMatrix();
  Assert(b->IsValid());
  Assert(fStatus & kDecomposed);
      
  if (fU.GetNrows() != b->GetNrows() || fU.GetRowLwb() != b->GetRowLwb())
  { 
    Error("Solve(TMatrixDColumn &cb","vector and matrix incompatible");
    return kFALSE; 
  }
      
  const Int_t n = fU.GetNrows();
    
  const Double_t *pU  = fU.GetMatrixArray();
        Double_t *pcb = cb.GetPtr();
  const Int_t     inc = cb.GetInc(); 
  
  Int_t i;
  // step 1: Forward substitution U^T
  for (i = 0; i < n; i++) { 
    const Int_t off_i  = i*n;
    const Int_t off_i2 = i*inc;
    if (pU[off_i+i] < fTol)
    {
      Error("Solve(TMatrixDColumn &cb)","u[%d,%d]=%.4e < %.4e",i,i,pU[off_i+i],fTol);
      return kFALSE;
    }
    Double_t r = pcb[off_i2];
    for (Int_t j = 0; j < i; j++) {
      const Int_t off_j = j*n;
      r -= pU[off_j+i]*pcb[j*inc];
    }
    pcb[off_i2] = r/pU[off_i+i];
  }

  // step 2: Backward substitution U
  for (i = n-1; i >= 0; i--) {
    const Int_t off_i  = i*n;
    const Int_t off_i2 = i*inc;
    Double_t r = pcb[off_i2];
    for (Int_t j = i+1; j < n; j++)
      r -= pU[off_i+j]*pcb[j*inc];
    pcb[off_i2] = r/pU[off_i+i];
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
  // Solve min {(A . x - b)^T (A . x - b)} for vector x where
  //   A : (m x n) matrix, m >= n
  //   b : (m)     vector
  //   x : (n)     vector

  TDecompChol ch(TMatrixDSym(TMatrixDBase::kAtA,A));
  Bool_t ok;
  return ch.Solve(TMatrixD(TMatrixDBase::kTransposed,A)*b,ok);
}

//______________________________________________________________________________
TVectorD NormalEqn(const TMatrixD &A,const TVectorD &b,const TVectorD &std)
{
  // Solve min {(A . x - b)^T W (A . x - b)} for vector x where
  //   A : (m x n) matrix, m >= n
  //   b : (m)     vector
  //   x : (n)     vector
  //   W : (m x m) weight matrix with W(i,j) = 1/std(i)^2  for i == j
  //                                         = 0           fir i != j

  if (!AreCompatible(b,std)) {
    ::Error("NormalEqn","vectors b and std are not compatible");
    return TVectorD();
  }

  TMatrixD Aw = A;
  TVectorD bw = b;
  for (Int_t irow = 0; irow < A.GetNrows(); irow++) {
    TMatrixDRow(Aw,irow) *= 1/std(irow);
    bw(irow) /= std(irow);
  }
  TDecompChol ch(TMatrixDSym(TMatrixDBase::kAtA,Aw));
  Bool_t ok;
  return ch.Solve(TMatrixD(TMatrixDBase::kTransposed,Aw)*bw,ok);
}

//______________________________________________________________________________
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B)
{
  // Solve min {(A . X_j - B_j)^T (A . X_j - B_j)} for each column j in
  // B and X
  //   A : (m x n ) matrix, m >= n
  //   B : (m x nb) matrix, nb >= 1
  //   X : (n x nb) matrix

  TDecompChol ch(TMatrixDSym(TMatrixDBase::kAtA,A));
  TMatrixD X(A,TMatrixDBase::kTransposeMult,B);
  ch.MultiSolve(X);
  return X;
}

//______________________________________________________________________________
TMatrixD NormalEqn(const TMatrixD &A,const TMatrixD &B,const TVectorD &std)
{
  // Solve min {(A . X_j - B_j)^T W (A . X_j - B_j)} for each column j in
  // B and X
  //   A : (m x n ) matrix, m >= n
  //   B : (m x nb) matrix, nb >= 1
  //   X : (n x nb) matrix
  //   W : (m x m) weight matrix with W(i,j) = 1/std(i)^2  for i == j
  //                                         = 0           fir i != j

  TMatrixD Aw = A;
  TMatrixD Bw = B;
  for (Int_t irow = 0; irow < A.GetNrows(); irow++) {
    TMatrixDRow(Aw,irow) *= 1/std(irow);
    TMatrixDRow(Bw,irow) *= 1/std(irow);
  }

  TDecompChol ch(TMatrixDSym(TMatrixDBase::kAtA,Aw));
  TMatrixD X(Aw,TMatrixDBase::kTransposeMult,Bw);
  ch.MultiSolve(X);
  return X;
}
