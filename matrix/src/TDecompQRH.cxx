// @(#)root/matrix:$Id$
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
//  fQ : (m x n) - orthogonal matrix                                     //
//  fR : (n x n) - upper triangular matrix                               //
//  H  : HouseHolder matrix which is stored through                      //
//  fUp: (n) - vector with Householder up's                              //
//  fW : (n) - vector with Householder beta's                            //
//                                                                       //
//  If row/column index of A starts at (rowLwb,colLwb) then              //
//  the decomposed matrices start from :                                 //
//  fQ  : (rowLwb,0)                                                     //
//  fR  : (0,colLwb)                                                     //
//  and the decomposed vectors start from :                              //
//  fUp : (0)                                                            //
//  fW  : (0)                                                            //
//                                                                       //
// Errors arise from formation of reflectors i.e. singularity .          //
// Note it attempts to handle the cases where the nRow <= nCol .         //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "TDecompQRH.h"

ClassImp(TDecompQRH)

//______________________________________________________________________________
TDecompQRH::TDecompQRH(Int_t nrows,Int_t ncols)
{
// Constructor for (nrows x ncols) matrix

   if (nrows < ncols) {
      Error("TDecompQRH(Int_t,Int_t","matrix rows should be >= columns");
      return;
   }

   fQ.ResizeTo(nrows,ncols);
   fR.ResizeTo(ncols,ncols);
   if (nrows <= ncols) {
      fW.ResizeTo(nrows);
      fUp.ResizeTo(nrows);
   } else {
      fW.ResizeTo(ncols);
      fUp.ResizeTo(ncols);
   }
}

//______________________________________________________________________________
TDecompQRH::TDecompQRH(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
// Constructor for ([row_lwb..row_upb] x [col_lwb..col_upb]) matrix

   const Int_t nrows = row_upb-row_lwb+1;
   const Int_t ncols = col_upb-col_lwb+1;

   if (nrows < ncols) {
      Error("TDecompQRH(Int_t,Int_t,Int_t,Int_t","matrix rows should be >= columns");
      return;
   }

   fRowLwb = row_lwb;
   fColLwb = col_lwb;

   fQ.ResizeTo(nrows,ncols);
   fR.ResizeTo(ncols,ncols);
   if (nrows <= ncols) {
      fW.ResizeTo(nrows);
      fUp.ResizeTo(nrows);
   } else {
      fW.ResizeTo(ncols);
      fUp.ResizeTo(ncols);
   }
}

//______________________________________________________________________________
TDecompQRH::TDecompQRH(const TMatrixD &a,Double_t tol)
{
// Constructor for general matrix A .

   R__ASSERT(a.IsValid());
   if (a.GetNrows() < a.GetNcols()) {
      Error("TDecompQRH(const TMatrixD &","matrix rows should be >= columns");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = a.Norm1();
   fTol = a.GetTol();
   if (tol > 0.0)
      fTol = tol;

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   const Int_t nRow = a.GetNrows();
   const Int_t nCol = a.GetNcols();

   fQ.ResizeTo(nRow,nCol);
   memcpy(fQ.GetMatrixArray(),a.GetMatrixArray(),nRow*nCol*sizeof(Double_t));
   fR.ResizeTo(nCol,nCol);
   if (nRow <= nCol) {
      fW.ResizeTo(nRow);
      fUp.ResizeTo(nRow);
   } else {
      fW.ResizeTo(nCol);
      fUp.ResizeTo(nCol);
   }
}

//______________________________________________________________________________
TDecompQRH::TDecompQRH(const TDecompQRH &another) : TDecompBase(another)
{
// Copy constructor

   *this = another;
}

//______________________________________________________________________________
Bool_t TDecompQRH::Decompose()
{
// QR decomposition of matrix a by Householder transformations,
//  see Golub & Loan first edition p41 & Sec 6.2.
// First fR is returned in upper triang of fQ and diagR. fQ returned in
// 'u-form' in lower triang of fQ and fW, the latter containing the
//  "Householder betas".
// If the decomposition succeeds, bit kDecomposed is set , otherwise kSingular

   if (TestBit(kDecomposed)) return kTRUE;

   if ( !TestBit(kMatrixSet) ) {
      Error("Decompose()","Matrix has not been set");
      return kFALSE;
   }

   const Int_t nRow   = this->GetNrows();
   const Int_t nCol   = this->GetNcols();
   const Int_t rowLwb = this->GetRowLwb();
   const Int_t colLwb = this->GetColLwb();

   TVectorD diagR;
   Double_t work[kWorkMax];
   if (nCol > kWorkMax) diagR.ResizeTo(nCol);
   else                 diagR.Use(nCol,work);

   if (QRH(fQ,diagR,fUp,fW,fTol)) {
      for (Int_t i = 0; i < nRow; i++) {
         const Int_t ic = (i < nCol) ? i : nCol;
         for (Int_t j = ic ; j < nCol; j++)
            fR(i,j) = fQ(i,j);
      }
      TMatrixDDiag diag(fR); diag = diagR;

      fQ.Shift(rowLwb,0);
      fR.Shift(0,colLwb);

      SetBit(kDecomposed);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::QRH(TMatrixD &q,TVectorD &diagR,TVectorD &up,TVectorD &w,Double_t tol)
{
// Decomposition function .

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
void TDecompQRH::SetMatrix(const TMatrixD &a)
{
// Set matrix to be decomposed

   R__ASSERT(a.IsValid());

   ResetStatus();
   if (a.GetNrows() < a.GetNcols()) {
      Error("TDecompQRH(const TMatrixD &","matrix rows should be >= columns");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = a.Norm1();

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   const Int_t nRow = a.GetNrows();
   const Int_t nCol = a.GetNcols();

   fQ.ResizeTo(nRow,nCol);
   memcpy(fQ.GetMatrixArray(),a.GetMatrixArray(),nRow*nCol*sizeof(Double_t));
   fR.ResizeTo(nCol,nCol);
   if (nRow <= nCol) {
      fW.ResizeTo(nRow);
      fUp.ResizeTo(nRow);
   } else {
      fW.ResizeTo(nCol);
      fUp.ResizeTo(nCol);
   }
}

//______________________________________________________________________________
Bool_t TDecompQRH::Solve(TVectorD &b)
{
// Solve Ax=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      Error("Solve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("Solve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fQ.GetNrows() != b.GetNrows() || fQ.GetRowLwb() != b.GetLwb()) {
      Error("Solve(TVectorD &","vector and matrix incompatible");
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

   const Double_t *pR = fR.GetMatrixArray();
         Double_t *pb = b.GetMatrixArray();

   // Backward substitution
   for (Int_t i = nRCol-1; i >= 0; i--) {
      const Int_t off_i = i*nRCol;
      Double_t r = pb[i];
      for (Int_t j = i+1; j < nRCol; j++)
         r -= pR[off_i+j]*pb[j];
      if (TMath::Abs(pR[off_i+i]) < fTol)
      {
         Error("Solve(TVectorD &)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
         return kFALSE;
      }
      pb[i] = r/pR[off_i+i];
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::Solve(TMatrixDColumn &cb)
{
// Solve Ax=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      Error("Solve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("Solve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fQ.GetNrows() != b->GetNrows() || fQ.GetRowLwb() != b->GetRowLwb())
   {
      Error("Solve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t nQRow = fQ.GetNrows();
   const Int_t nQCol = fQ.GetNcols();

   // Calculate  Q^T.b
   const Int_t nQ = (nQRow <= nQCol) ? nQRow-1 : nQCol;
   for (Int_t k = 0; k < nQ; k++) {
      const TVectorD qc_k = TMatrixDColumn_const(fQ,k);
      ApplyHouseHolder(qc_k,fUp(k),fW(k),k,k+1,cb);
   }

   const Int_t nRCol = fR.GetNcols();

   const Double_t *pR  = fR.GetMatrixArray();
         Double_t *pcb = cb.GetPtr();
   const Int_t     inc = cb.GetInc();

   // Backward substitution
   for (Int_t i = nRCol-1; i >= 0; i--) {
      const Int_t off_i  = i*nRCol;
      const Int_t off_i2 = i*inc;
      Double_t r = pcb[off_i2];
      for (Int_t j = i+1; j < nRCol; j++)
         r -= pR[off_i+j]*pcb[j*inc];
      if (TMath::Abs(pR[off_i+i]) < fTol)
      {
         Error("Solve(TMatrixDColumn &)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
         return kFALSE;
      }
      pcb[off_i2] = r/pR[off_i+i];
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::TransSolve(TVectorD &b)
{
// Solve A^T x=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      Error("TransSolve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("TransSolve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fQ.GetNrows() != fQ.GetNcols() || fQ.GetRowLwb() != fQ.GetColLwb()) {
      Error("TransSolve(TVectorD &","matrix should be square");
      return kFALSE;
   }

   if (fR.GetNrows() != b.GetNrows() || fR.GetRowLwb() != b.GetLwb()) {
      Error("TransSolve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   const Double_t *pR = fR.GetMatrixArray();
         Double_t *pb = b.GetMatrixArray();

   const Int_t nRCol = fR.GetNcols();

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
         Error("TransSolve(TVectorD &)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
         return kFALSE;
      }
      pb[i] = r/pR[off_i+i];
   }

   const Int_t nQRow = fQ.GetNrows();

   // Calculate  Q.b; it was checked nQRow == nQCol
   for (Int_t k = nQRow-1; k >= 0; k--) {
      const TVectorD qc_k = TMatrixDColumn_const(fQ,k);
      ApplyHouseHolder(qc_k,fUp(k),fW(k),k,k+1,b);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompQRH::TransSolve(TMatrixDColumn &cb)
{
// Solve A^T x=b assuming the QR form of A is stored in fR,fQ and fW, but assume b
// has *not* been transformed.  Solution returned in b.

   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      Error("TransSolve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("TransSolve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fQ.GetNrows() != fQ.GetNcols() || fQ.GetRowLwb() != fQ.GetColLwb()) {
      Error("TransSolve(TMatrixDColumn &","matrix should be square");
      return kFALSE;
   }

   if (fR.GetNrows() != b->GetNrows() || fR.GetRowLwb() != b->GetRowLwb()) {
      Error("TransSolve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   const Double_t *pR  = fR.GetMatrixArray();
         Double_t *pcb = cb.GetPtr();
   const Int_t     inc = cb.GetInc();

   const Int_t nRCol = fR.GetNcols();

   // Backward substitution
   for (Int_t i = 0; i < nRCol; i++) {
      const Int_t off_i  = i*nRCol;
      const Int_t off_i2 = i*inc;
      Double_t r = pcb[off_i2];
      for (Int_t j = 0; j < i; j++) {
         const Int_t off_j = j*nRCol;
         r -= pR[off_j+i]*pcb[j*inc];
      }
      if (TMath::Abs(pR[off_i+i]) < fTol)
      {
         Error("TransSolve(TMatrixDColumn &)","R[%d,%d]=%.4e < %.4e",i,i,pR[off_i+i],fTol);
         return kFALSE;
      }
      pcb[off_i2] = r/pR[off_i+i];
   }

   const Int_t nQRow = fQ.GetNrows();

   // Calculate  Q.b; it was checked nQRow == nQCol
   for (Int_t k = nQRow-1; k >= 0; k--) {
      const TVectorD qc_k = TMatrixDColumn_const(fQ,k);
      ApplyHouseHolder(qc_k,fUp(k),fW(k),k,k+1,cb);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TDecompQRH::Det(Double_t &d1,Double_t &d2)
{
// This routine calculates the absolute (!) value of the determinant
// |det| = d1*TMath::Power(2.,d2)

   if ( !TestBit(kDetermined) ) {
      if ( !TestBit(kDecomposed) )
        Decompose();
      if (TestBit(kSingular)) {
         fDet1 = 0.0;
         fDet2 = 0.0;
      } else
         TDecompBase::Det(d1,d2);
      SetBit(kDetermined);
   }
   d1 = fDet1;
   d2 = fDet2;
}

//______________________________________________________________________________
Bool_t TDecompQRH::Invert(TMatrixD &inv)
{
// For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
// The user should always supply a matrix of size (m x m) !
// If m > n , only the (n x m) part of the returned (pseudo inverse) matrix
// should be used .

   if (inv.GetNrows()  != GetNrows()  || inv.GetNcols()  != GetNrows() ||
       inv.GetRowLwb() != GetRowLwb() || inv.GetColLwb() != GetColLwb()) {
      Error("Invert(TMatrixD &","Input matrix has wrong shape");
      return kFALSE;
   }

   inv.UnitMatrix();
   const Bool_t status = MultiSolve(inv);

   return status;
}

//______________________________________________________________________________
TMatrixD TDecompQRH::Invert(Bool_t &status)
{
// For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
// (n x m) Ainv is returned .

   const Int_t rowLwb = GetRowLwb();
   const Int_t colLwb = GetColLwb();
   const Int_t rowUpb = rowLwb+GetNrows()-1;
   TMatrixD inv(rowLwb,rowUpb,colLwb,colLwb+GetNrows()-1);
   inv.UnitMatrix();
   status = MultiSolve(inv);
   inv.ResizeTo(rowLwb,rowLwb+GetNcols()-1,colLwb,colLwb+GetNrows()-1);

   return inv;
}

//______________________________________________________________________________
void TDecompQRH::Print(Option_t *opt) const
{
// Print the class members

   TDecompBase::Print(opt);
   fQ.Print("fQ");
   fR.Print("fR");
   fUp.Print("fUp");
   fW.Print("fW");
}

//______________________________________________________________________________
TDecompQRH &TDecompQRH::operator=(const TDecompQRH &source)
{
// Assignment operator

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
