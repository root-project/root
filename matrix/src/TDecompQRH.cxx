// @(#)root/matrix:$Name:  $:$Id: TDecompQRH.cxx,v 1.47 2003/09/05 09:21:54 brun Exp $
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
// QR Decomposition class                                                //
//                                                                       //
// Decompose  a general (m x n) matrix A into A = fQ fR H   where        //
//                                                                       //
//  fR : (n x n) - upper triangular matrix                               //
//  fQ : (m x n) - orthogonal matrix                                     //
//  H  : HouseHolder matrix which is stored through                      //
//  fUp: n - vector with Householder up's                                //
//  fW : n - vector with Householder beta's                              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "TDecompQRH.h"

ClassImp(TDecompQRH)

//______________________________________________________________________________
TDecompQRH::TDecompQRH(const TMatrixD &a,Double_t tol)
{
  Assert(a.IsValid());

  fCondition = a.Norm1();
  fTol = a.GetTol();
  if (tol > 0.0)
    fTol = tol;

  Decompose(a);
}

//______________________________________________________________________________
TDecompQRH::TDecompQRH(const TDecompQRH &another) : TDecompBase(another)
{
  *this = another;
}

//______________________________________________________________________________
Int_t TDecompQRH::Decompose(const TMatrixDBase &a)
{
// QR decomposition of a by Householder transformations. See Golub & Loan
// first edition p41 & Sec 6.2.
// First fR is returned in upper triang of fQ and diagR. fQ returned in
// 'u-form' in lower triang of fQ and fW, the latter containing the
//  "Householder betas".
// Errors arise from formation of reflectors i.e. singularity .
// Note it attempts to handle the cases where the nRow <= nCol .

  Assert(a.IsValid());

  const Int_t nRow   = a.GetNrows();
  const Int_t nCol   = a.GetNcols();
  const Int_t rowLwb = a.GetRowLwb();
  const Int_t colLwb = a.GetColLwb();

  fQ.ResizeTo(nRow,nCol);
  memcpy(fQ.GetElements(),a.GetElements(),nRow*nCol*sizeof(Double_t));
  fR.ResizeTo(nCol,nCol);
  if (nRow <= nCol) {
    fW.ResizeTo(nRow);
    fUp.ResizeTo(nRow);
  } else {
    fW.ResizeTo(nCol);
    fUp.ResizeTo(nCol);
  }

  TVectorD diagR(nCol);

  if (QRH(fQ,diagR,fUp,fW,fTol)) {
    for (Int_t i = 0; i < nRow; i++) {
      const Int_t ic = (i < nCol) ? i : nCol;
      for (Int_t j = ic ; j < nCol; j++)
        fR(i,j) = fQ(i,j);
    }
    TMatrixDDiag(fR,0) = diagR;

    fQ.Shift(rowLwb,0);
    fR.Shift(0,colLwb);

    fStatus |= kDecomposed;
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::QRH(TMatrixD &q,TVectorD &diagR,TVectorD &up,TVectorD &w,Double_t tol)
{
  const Int_t nRow = q.GetNrows();
  const Int_t nCol = q.GetNcols();

  const Int_t n = (nRow <= nCol) ? nRow-1 : nCol;

  for (Int_t k = 0 ; k < n ; k++) {
    const TVectorD qc_k = TMatrixDColumn_const(q,k);
    if (!DefHouseHolder(qc_k,k,k+1,up(k),w(k),tol))
      return kFALSE;
    diagR(k) = qc_k(k)-up(k);
    if (k < nCol-1) {
      // Apply HouseHolder to sub-matrix
      for (Int_t j = k+1; j < nCol; j++) {
        TMatrixDColumn qc_j = TMatrixDColumn(q,j);
        ApplyHouseHolder(qc_k,up(k),w(k),k,k+1,qc_j);
      }
    }
  }

  if (nRow <= nCol) {
    diagR(nRow-1) = q(nRow-1,nRow-1);
    up(nRow-1) = 0;
    w(nRow-1)  = 0;
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::Solve(TVectorD &b)
{
// Solve Ax=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

  Assert(b.IsValid());
  Assert(fR.IsValid() && fQ.IsValid() && fW.IsValid());

  if (fQ.GetNrows() != b.GetNrows() || fQ.GetRowLwb() != b.GetLwb())
  { 
    Error("Solve(TVectorD &","vector and matrix incompatible");
    b.Invalidate();
    return kFALSE;
  } 

  const Int_t nQRow = fQ.GetNrows();
  const Int_t nQCol = fQ.GetNcols();

  // Calculate  Q^T.b
  const Int_t nQ = (nQRow <= nQCol) ? nQRow-1 : nQCol;
  for (Int_t k = 0; k < nQ; k++) {
    const TVectorD qc_k = TMatrixDColumn_const(fQ,k);
    ApplyHouseHolder(qc_k,fUp(k),fW(k),k,k+1,b);
  }

  const Int_t nRCol = fR.GetNcols();
  Assert(b.GetNrows() >= nRCol);

  const Double_t *pR = fR.GetElements();
        Double_t *pb = b.GetElements();

  // Backward substitution
  for (Int_t i = nRCol-1; i >= 0; i--) {
    const Int_t off_i = i*nRCol;
    Double_t r = pb[i];
    for (Int_t j = i+1; j < nRCol; j++)
      r -= pR[off_i+j]*pb[j];
    if (TMath::Abs(pR[off_i+i]) < fTol)
    {
      Error("Solve(TVectorD &b)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
      return kFALSE;
    }
    pb[i] = r/pR[off_i+i];
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::Solve(TMatrixDColumn &cb)
{
  const TMatrixDBase *b = cb.GetMatrix();
  Assert(b->IsValid());
  Assert(fR.IsValid() && fQ.IsValid() && fW.IsValid());

  if (fR.GetNrows() != b->GetNrows() || fR.GetRowLwb() != b->GetRowLwb())
  { 
    Error("Solve(TMatrixDColumn &","vector and matrix incompatible");
    return kFALSE; 
  }     

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::TransSolve(TVectorD &b)
{
// Solve A^T x=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

  Assert(b.IsValid());
  Assert(fR.IsValid() && fQ.IsValid() && fW.IsValid());

  if (fQ.GetNrows() != b.GetNrows() || fQ.GetRowLwb() != b.GetLwb())
  {   
    Error("TransSolve(TVectorD &","vector and matrix incompatible");
    b.Invalidate();
    return kFALSE;
  } 

  const Int_t nRCol = fR.GetNcols();
  Assert(b.GetNrows() >= nRCol);

  const Double_t *pR = fR.GetElements();
        Double_t *pb = b.GetElements();

  // Backward substitution
  for (Int_t i = 0; i < nRCol; i++) {
    const Int_t off_i = i*nRCol;
    Double_t r = pb[i];
    for (Int_t j = 0; j < i; j++) {
      const Int_t off_j = j*nRCol;
      r -= pR[off_j+i]*pb[j];
    }
    if (TMath::Abs(pR[off_i+i]) < fTol)
    {
      Error("TransSolve(TVectorD &b)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
      return kFALSE;
    }
    pb[i] = r/pR[off_i+i];
  }

  const Int_t nQRow = fQ.GetNrows();
  const Int_t nQCol = fQ.GetNcols();

  Assert(nQRow == nQCol);

  // Calculate  Q.b
  for (Int_t k = nQRow-1; k >= 0; k--) {
    const TVectorD qc_k = TMatrixDColumn_const(fQ,k);
    ApplyHouseHolder(qc_k,fUp(k),fW(k),k,k+1,b);
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::TransSolve(TMatrixDColumn &cb)
{
  const TMatrixDBase *b = cb.GetMatrix();
  Assert(b->IsValid());
  Assert(fR.IsValid() && fQ.IsValid() && fW.IsValid());

  if (fR.GetNrows() != b->GetNrows() || fR.GetRowLwb() != b->GetRowLwb())
  {
    Error("TransSolve(TMatrixDColumn &","vector and matrix incompatible");
    return kFALSE; 
  }   

  return kTRUE;
}

//______________________________________________________________________________
void TDecompQRH::Det(Double_t &d1,Double_t &d2)
{
  if ( !( fStatus & kDetermined ) ) {
    if ( fStatus & kSingular ) {
      fDet1 = 0.0;
      fDet2 = 0.0;
    } else
      TDecompBase::Det(d1,d2);
    fStatus |= kDetermined;
  }
  d1 = fDet1;
  d2 = fDet2;
}

//______________________________________________________________________________
TDecompQRH &TDecompQRH::operator=(const TDecompQRH &source)
{ 
  if (this != &source) {
    TDecompBase::operator=(source);
    fQ.ResizeTo(source.fQ);
    fR.ResizeTo(source.fR);
    fUp.ResizeTo(source.fUp);
    fW.ResizeTo(source.fW);
    fQ  = source.fQ;
    fR  = source.fR;
    fUp = source.fUp;
    fW  = source.fW;
  }
  return *this;
}     
