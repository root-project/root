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
// Single Value Decomposition class                                      //
//                                                                       //
// For an (m x n) matrix A with m >= n, the singular value decomposition //
// is                                                                    //
// an (m x m) orthogonal matrix fU, an (m x n) diagonal matrix fS, and   //
// an (n x n) orthogonal matrix fV so that A = U*S*V'.                   //
//                                                                       //
// If the row/column index of A starts at (rowLwb,colLwb) then the       //
// decomposed matrices/vectors start at :                                //
//  fU   : (rowLwb,colLwb)                                               //
//  fV   : (colLwb,colLwb)                                               //
//  fSig : (colLwb)                                                      //
//                                                                       //
// The diagonal matrix fS is stored in the singular values vector fSig . //
// The singular values, fSig[k] = S[k][k], are ordered so that           //
// fSig[0] >= fSig[1] >= ... >= fSig[n-1].                               //
//                                                                       //
// The singular value decompostion always exists, so the decomposition   //
// will (as long as m >=n) never fail. If m < n, the user should add     //
// sufficient zero rows to A , so that m == n                            //
//                                                                       //
// Here fTol is used to set the threshold on the minimum allowed value   //
// of the singular values:                                               //
//  min_singular = fTol*max(fSig[i])                                     //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "TDecompSVD.h"
#include "TMath.h"
#include "TArrayD.h"

ClassImp(TDecompSVD)

//______________________________________________________________________________
TDecompSVD::TDecompSVD(Int_t nrows,Int_t ncols)
{
// Constructor for (nrows x ncols) matrix

   if (nrows < ncols) {
      Error("TDecompSVD(Int_t,Int_t","matrix rows should be >= columns");
      return;
   }
   fU.ResizeTo(nrows,nrows);
   fSig.ResizeTo(ncols);
   fV.ResizeTo(nrows,ncols); // In the end we only need the nColxnCol part
}

//______________________________________________________________________________
TDecompSVD::TDecompSVD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
{
// Constructor for ([row_lwb..row_upb] x [col_lwb..col_upb]) matrix

   const Int_t nrows = row_upb-row_lwb+1;
   const Int_t ncols = col_upb-col_lwb+1;

   if (nrows < ncols) {
      Error("TDecompSVD(Int_t,Int_t,Int_t,Int_t","matrix rows should be >= columns");
      return;
   }
   fRowLwb = row_lwb;
   fColLwb = col_lwb;
   fU.ResizeTo(nrows,nrows);
   fSig.ResizeTo(ncols);
   fV.ResizeTo(nrows,ncols); // In the end we only need the nColxnCol part
}

//______________________________________________________________________________
TDecompSVD::TDecompSVD(const TMatrixD &a,Double_t tol)
{
// Constructor for general matrix A .

   R__ASSERT(a.IsValid());
   if (a.GetNrows() < a.GetNcols()) {
      Error("TDecompSVD(const TMatrixD &","matrix rows should be >= columns");
      return;
   }

   SetBit(kMatrixSet);
   fTol = a.GetTol();
   if (tol > 0)
      fTol = tol;

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   const Int_t nRow = a.GetNrows();
   const Int_t nCol = a.GetNcols();

   fU.ResizeTo(nRow,nRow);
   fSig.ResizeTo(nCol);
   fV.ResizeTo(nRow,nCol); // In the end we only need the nColxnCol part

   fU.UnitMatrix();
   memcpy(fV.GetMatrixArray(),a.GetMatrixArray(),nRow*nCol*sizeof(Double_t));
}

//______________________________________________________________________________
TDecompSVD::TDecompSVD(const TDecompSVD &another): TDecompBase(another)
{
// Copy constructor

   *this = another;
}

//______________________________________________________________________________
Bool_t TDecompSVD::Decompose()
{
// SVD decomposition of matrix
// If the decomposition succeeds, bit kDecomposed is set , otherwise kSingular

   if (TestBit(kDecomposed)) return kTRUE;

   if ( !TestBit(kMatrixSet) ) {
      Error("Decompose()","Matrix has not been set");
      return kFALSE;
   }

   const Int_t nCol   = this->GetNcols();
   const Int_t rowLwb = this->GetRowLwb();
   const Int_t colLwb = this->GetColLwb();

   TVectorD offDiag;
   Double_t work[kWorkMax];
   if (nCol > kWorkMax) offDiag.ResizeTo(nCol);
   else                 offDiag.Use(nCol,work);

   // step 1: bidiagonalization of A
   if (!Bidiagonalize(fV,fU,fSig,offDiag))
      return kFALSE;

   // step 2: diagonalization of bidiagonal matrix
   if (!Diagonalize(fV,fU,fSig,offDiag))
      return kFALSE;

   // step 3: order singular values and perform permutations
   SortSingular(fV,fU,fSig);
   fV.ResizeTo(nCol,nCol); fV.Shift(colLwb,colLwb);
   fSig.Shift(colLwb);
   fU.Transpose(fU);       fU.Shift(rowLwb,colLwb);
   SetBit(kDecomposed);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompSVD::Bidiagonalize(TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag)
{
// Bidiagonalize the (m x n) - matrix a (stored in v) through a series of Householder
// transformations applied to the left (Q^T) and to the right (H) of a ,
// so that A = Q . C . H^T with matrix C bidiagonal. Q and H are orthogonal matrices .
//
// Output:
//   v     - (n x n) - matrix H in the (n x n) part of v
//   u     - (m x m) - matrix Q^T
//   sDiag - diagonal of the (m x n) C
//   oDiag - off-diagonal elements of matrix C
//
//  Test code for the output:
//    const Int_t nRow = v.GetNrows();
//    const Int_t nCol = v.GetNcols();
//    TMatrixD H(v); H.ResizeTo(nCol,nCol);
//    TMatrixD E1(nCol,nCol); E1.UnitMatrix();
//    TMatrixD Ht(TMatrixDBase::kTransposed,H);
//    Bool_t ok = kTRUE;
//    ok &= VerifyMatrixIdentity(Ht * H,E1,kTRUE,1.0e-13);
//    ok &= VerifyMatrixIdentity(H * Ht,E1,kTRUE,1.0e-13);
//    TMatrixD E2(nRow,nRow); E2.UnitMatrix();
//    TMatrixD Qt(u);
//    TMatrixD Q(TMatrixDBase::kTransposed,Qt);
//    ok &= VerifyMatrixIdentity(Q * Qt,E2,kTRUE,1.0e-13);
//    TMatrixD C(nRow,nCol);
//    TMatrixDDiag(C) = sDiag;
//    for (Int_t i = 0; i < nCol-1; i++)
//      C(i,i+1) = oDiag(i+1);
//    TMatrixD A = Q*C*Ht;
//    ok &= VerifyMatrixIdentity(A,a,kTRUE,1.0e-13);

   const Int_t nRow_v = v.GetNrows();
   const Int_t nCol_v = v.GetNcols();
   const Int_t nCol_u = u.GetNcols();

   TArrayD ups(nCol_v);
   TArrayD betas(nCol_v);

   for (Int_t i = 0; i < nCol_v; i++) {
      // Set up Householder Transformation q(i)

      Double_t up,beta;
      if (i < nCol_v-1 || nRow_v > nCol_v) {
         const TVectorD vc_i = TMatrixDColumn_const(v,i);
         //if (!DefHouseHolder(vc_i,i,i+1,up,beta))
         //  return kFALSE;
         DefHouseHolder(vc_i,i,i+1,up,beta);

         // Apply q(i) to v
         for (Int_t j = i; j < nCol_v; j++) {
            TMatrixDColumn vc_j = TMatrixDColumn(v,j);
            ApplyHouseHolder(vc_i,up,beta,i,i+1,vc_j);
         }

         // Apply q(i) to u
         for (Int_t j = 0; j < nCol_u; j++)
         {
            TMatrixDColumn uc_j = TMatrixDColumn(u,j);
            ApplyHouseHolder(vc_i,up,beta,i,i+1,uc_j);
         }
      }
      if (i < nCol_v-2) {
         // set up Householder Transformation h(i)
         const TVectorD vr_i = TMatrixDRow_const(v,i);

         //if (!DefHouseHolder(vr_i,i+1,i+2,up,beta))
         //  return kFALSE;
         DefHouseHolder(vr_i,i+1,i+2,up,beta);

         // save h(i)
         ups[i]   = up;
         betas[i] = beta;

         // apply h(i) to v
         for (Int_t j = i; j < nRow_v; j++) {
            TMatrixDRow vr_j = TMatrixDRow(v,j);
            ApplyHouseHolder(vr_i,up,beta,i+1,i+2,vr_j);

            // save elements i+2,...in row j of matrix v
            if (j == i) {
               for (Int_t k = i+2; k < nCol_v; k++)
                  vr_j(k) = vr_i(k);
            }
         }
      }
   }

   // copy diagonal of transformed matrix v to sDiag and upper parallel v to oDiag
   if (nCol_v > 1) {
      for (Int_t i = 1; i < nCol_v; i++)
         oDiag(i) = v(i-1,i);
   }
   oDiag(0) = 0.;
   sDiag = TMatrixDDiag(v);

   // construct product matrix h = h(1)*h(2)*...*h(nCol_v-1), h(nCol_v-1) = I

   TVectorD vr_i(nCol_v);
   for (Int_t i = nCol_v-1; i >= 0; i--) {
      if (i < nCol_v-1)
         vr_i = TMatrixDRow_const(v,i);
      TMatrixDRow(v,i) = 0.0;
      v(i,i) = 1.;

      if (i < nCol_v-2) {
         for (Int_t k = i; k < nCol_v; k++) {
            // householder transformation on k-th column
            TMatrixDColumn vc_k = TMatrixDColumn(v,k);
            ApplyHouseHolder(vr_i,ups[i],betas[i],i+1,i+2,vc_k);
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompSVD::Diagonalize(TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag)
{
// Diagonalizes in an iterative fashion the bidiagonal matrix C as described through
// sDiag and oDiag, so that S' = U'^T . C . V' is diagonal. U' and V' are orthogonal
// matrices .
//
// Output:
//   v     - (n x n) - matrix H . V' in the (n x n) part of v
//   u     - (m x m) - matrix U'^T . Q^T
//   sDiag - diagonal of the (m x n) S'
//
//   return convergence flag:  0 -> no convergence
//                             1 -> convergence
//
//  Test code for the output:
//    const Int_t nRow = v.GetNrows();
//    const Int_t nCol = v.GetNcols();
//    TMatrixD tmp = v; tmp.ResizeTo(nCol,nCol);
//    TMatrixD Vprime  = Ht*tmp;
//    TMatrixD Vprimet(TMatrixDBase::kTransposed,Vprime);
//    TMatrixD Uprimet = u*Q;
//    TMatrixD Uprime(TMatrixDBase::kTransposed,Uprimet);
//    TMatrixD Sprime(nRow,nCol);
//    TMatrixDDiag(Sprime) = sDiag;
//    ok &= VerifyMatrixIdentity(Uprimet * C * Vprime,Sprime,kTRUE,1.0e-13);
//    ok &= VerifyMatrixIdentity(Q*Uprime * Sprime * Vprimet * Ht,a,kTRUE,1.0e-13);

   Bool_t ok    = kTRUE;
   Int_t niter  = 0;
   Double_t bmx = sDiag(0);

   const Int_t nCol_v = v.GetNcols();

   if (nCol_v > 1) {
      for (Int_t i = 1; i < nCol_v; i++)
         bmx = TMath::Max(TMath::Abs(sDiag(i))+TMath::Abs(oDiag(i)),bmx);
   }

   const Double_t eps = std::numeric_limits<double>::epsilon();

   const Int_t niterm = 10*nCol_v;
   for (Int_t k = nCol_v-1; k >= 0; k--) {
      loop:
         if (k != 0) {
            // since sDiag(k) == 0 perform Givens transform with result oDiag[k] = 0
            if (TMath::Abs(sDiag(k)) < eps*bmx)
               Diag_1(v,sDiag,oDiag,k);

            // find l (1 <= l <=k) so that either oDiag(l) = 0 or sDiag(l-1) = 0.
            // In the latter case transform oDiag(l) to zero. In both cases the matrix
            // splits and the bottom right minor begins with row l.
            // If no such l is found set l = 1.

            Int_t elzero = 0;
            Int_t l = 0;
            for (Int_t ll = k; ll >= 0 ; ll--) {
               l = ll;
               if (l == 0) {
                  elzero = 0;
                  break;
               } else if (TMath::Abs(oDiag(l)) < eps*bmx) {
                  elzero = 1;
                  break;
               } else if (TMath::Abs(sDiag(l-1)) < eps*bmx)
                  elzero = 0;
            }
            if (l > 0 && !elzero)
               Diag_2(sDiag,oDiag,k,l);
            if (l != k) {
               // one more QR pass with order k
               Diag_3(v,u,sDiag,oDiag,k,l);
               niter++;
               if (niter <= niterm) goto loop;
               ::Error("TDecompSVD::Diagonalize","no convergence after %d steps",niter);
               ok = kFALSE;
            }
         }

         if (sDiag(k) < 0.) {
            // for negative singular values perform change of sign
            sDiag(k) = -sDiag(k);
            TMatrixDColumn(v,k) *= -1.0;
         }
      // order is decreased by one in next pass
   }

   return ok;
}

//______________________________________________________________________________
void TDecompSVD::Diag_1(TMatrixD &v,TVectorD &sDiag,TVectorD &oDiag,Int_t k)
{
// Step 1 in the matrix diagonalization

   const Int_t nCol_v = v.GetNcols();

   TMatrixDColumn vc_k = TMatrixDColumn(v,k);
   for (Int_t i = k-1; i >= 0; i--) {
      TMatrixDColumn vc_i = TMatrixDColumn(v,i);
      Double_t h,cs,sn;
      if (i == k-1)
         DefAplGivens(sDiag[i],oDiag[i+1],cs,sn);
      else
         DefAplGivens(sDiag[i],h,cs,sn);
      if (i > 0) {
         h = 0.;
         ApplyGivens(oDiag[i],h,cs,sn);
      }
      for (Int_t j = 0; j < nCol_v; j++)
         ApplyGivens(vc_i(j),vc_k(j),cs,sn);
   }
}

//______________________________________________________________________________
void TDecompSVD::Diag_2(TVectorD &sDiag,TVectorD &oDiag,Int_t k,Int_t l)
{
// Step 2 in the matrix diagonalization

   for (Int_t i = l; i <= k; i++) {
      Double_t h,cs,sn;
      if (i == l)
         DefAplGivens(sDiag(i),oDiag(i),cs,sn);
      else
         DefAplGivens(sDiag(i),h,cs,sn);
      if (i < k) {
         h = 0.;
         ApplyGivens(oDiag(i+1),h,cs,sn);
      }
   }
}

//______________________________________________________________________________
void TDecompSVD::Diag_3(TMatrixD &v,TMatrixD &u,TVectorD &sDiag,TVectorD &oDiag,Int_t k,Int_t l)
{
// Step 3 in the matrix diagonalization

   Double_t *pS = sDiag.GetMatrixArray();
   Double_t *pO = oDiag.GetMatrixArray();

   // determine shift parameter

   const Double_t psk1 = pS[k-1];
   const Double_t psk  = pS[k];
   const Double_t pok1 = pO[k-1];
   const Double_t pok  = pO[k];
   const Double_t psl  = pS[l];

   Double_t f;
   if (psl == 0.0 || pok == 0.0 || psk1 == 0.0) {
      const Double_t b = ((psk1-psk)*(psk1+psk)+pok1*pok1)/2.0;
      const Double_t c = (psk*pok1)*(psk*pok1);

      Double_t shift = 0.0;
      if ((b != 0.0) | (c != 0.0)) {
         shift = TMath::Sqrt(b*b+c);
         if (b < 0.0)
            shift = -shift;
         shift = c/(b+shift);
      }

      f = (psl+psk)*(psl-psk)+shift;
   } else {
      f = ((psk1-psk)*(psk1+psk)+(pok1-pok)*(pok1+pok))/(2.*pok*psk1);
      const Double_t g = TMath::Hypot(1.,f);
      const Double_t t = (f >= 0.) ? f+g : f-g;

      f = ((psl-psk)*(psl+psk)+pok*(psk1/t-pok))/psl;
   }

   const Int_t nCol_v = v.GetNcols();
   const Int_t nCol_u = u.GetNcols();

   Double_t h,cs,sn;
   Int_t j;
   for (Int_t i = l; i <= k-1; i++) {
      if (i == l)
         // define r[l]
         DefGivens(f,pO[i+1],cs,sn);
      else
         // define r[i]
         DefAplGivens(pO[i],h,cs,sn);

      ApplyGivens(pS[i],pO[i+1],cs,sn);
      h = 0.;
      ApplyGivens(h,pS[i+1],cs,sn);

      TMatrixDColumn vc_i  = TMatrixDColumn(v,i);
      TMatrixDColumn vc_i1 = TMatrixDColumn(v,i+1);
      for (j = 0; j < nCol_v; j++)
         ApplyGivens(vc_i(j),vc_i1(j),cs,sn);
      // define t[i]
      DefAplGivens(pS[i],h,cs,sn);
      ApplyGivens(pO[i+1],pS[i+1],cs,sn);
      if (i < k-1) {
         h = 0.;
         ApplyGivens(h,pO[i+2],cs,sn);
      }

      TMatrixDRow ur_i  = TMatrixDRow(u,i);
      TMatrixDRow ur_i1 = TMatrixDRow(u,i+1);
      for (j = 0; j < nCol_u; j++)
         ApplyGivens(ur_i(j),ur_i1(j),cs,sn);
   }
}

//______________________________________________________________________________
void TDecompSVD::SortSingular(TMatrixD &v,TMatrixD &u,TVectorD &sDiag)
{
// Perform a permutation transformation on the diagonal matrix S', so that
// matrix S'' = U''^T . S' . V''  has diagonal elements ordered such that they
// do not increase.
//
// Output:
//   v     - (n x n) - matrix H . V' . V'' in the (n x n) part of v
//   u     - (m x m) - matrix U''^T . U'^T . Q^T
//   sDiag - diagonal of the (m x n) S''

   const Int_t nCol_v = v.GetNcols();
   const Int_t nCol_u = u.GetNcols();

   Double_t *pS = sDiag.GetMatrixArray();
   Double_t *pV = v.GetMatrixArray();
   Double_t *pU = u.GetMatrixArray();

   // order singular values

   Int_t i,j;
   if (nCol_v > 1) {
      while (1) {
         Bool_t found = kFALSE;
         i = 1;
         while (!found && i < nCol_v) {
            if (pS[i] > pS[i-1])
               found = kTRUE;
            else
               i++;
         }
         if (!found) break;
         for (i = 1; i < nCol_v; i++) {
            Double_t t = pS[i-1];
            Int_t k = i-1;
            for (j = i; j < nCol_v; j++) {
               if (t < pS[j]) {
                  t = pS[j];
                  k = j;
               }
            }
            if (k != i-1) {
               // perform permutation on singular values
               pS[k]   = pS[i-1];
               pS[i-1] = t;
               // perform permutation on matrix v
               for (j = 0; j < nCol_v; j++) {
                  const Int_t off_j = j*nCol_v;
                  t             = pV[off_j+k];
                  pV[off_j+k]   = pV[off_j+i-1];
                  pV[off_j+i-1] = t;
               }
               // perform permutation on vector u
               for (j = 0; j < nCol_u; j++) {
                  const Int_t off_k  = k*nCol_u;
                  const Int_t off_i1 = (i-1)*nCol_u;
                  t            = pU[off_k+j];
                  pU[off_k+j]  = pU[off_i1+j];
                  pU[off_i1+j] = t;
               }
            }
         }
      }
   }
}

//______________________________________________________________________________
const TMatrixD TDecompSVD::GetMatrix()
{
// Reconstruct the original matrix using the decomposition parts

   if (TestBit(kSingular)) {
      Error("GetMatrix()","Matrix is singular");
      return TMatrixD();
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("GetMatrix()","Decomposition failed");
         return TMatrixD();
      }
   }

   const Int_t nRows  = fU.GetNrows();
   const Int_t nCols  = fV.GetNcols();
   const Int_t colLwb = this->GetColLwb();
   TMatrixD s(nRows,nCols); s.Shift(colLwb,colLwb);
   TMatrixDDiag diag(s); diag = fSig;
   const TMatrixD vt(TMatrixD::kTransposed,fV);
   return fU * s * vt;
}

//______________________________________________________________________________
void TDecompSVD::SetMatrix(const TMatrixD &a)
{
// Set matrix to be decomposed

   R__ASSERT(a.IsValid());

   ResetStatus();
   if (a.GetNrows() < a.GetNcols()) {
      Error("TDecompSVD(const TMatrixD &","matrix rows should be >= columns");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = -1.0;

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   const Int_t nRow = a.GetNrows();
   const Int_t nCol = a.GetNcols();

   fU.ResizeTo(nRow,nRow);
   fSig.ResizeTo(nCol);
   fV.ResizeTo(nRow,nCol); // In the end we only need the nColxnCol part

   fU.UnitMatrix();
   memcpy(fV.GetMatrixArray(),a.GetMatrixArray(),nRow*nCol*sizeof(Double_t));
}

//______________________________________________________________________________
Bool_t TDecompSVD::Solve(TVectorD &b)
{
// Solve Ax=b assuming the SVD form of A is stored . Solution returned in b.
// If A is of size (m x n), input vector b should be of size (m), however,
// the solution, returned in b, will be in the first (n) elements .
//
// For m > n , x  is the least-squares solution of min(A . x - b)

   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         return kFALSE;
      }
   }

   if (fU.GetNrows() != b.GetNrows() || fU.GetRowLwb() != b.GetLwb())
   {
      Error("Solve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   // We start with fU fSig fV^T x = b, and turn it into  fV^T x = fSig^-1 fU^T b
   // Form tmp = fSig^-1 fU^T b but ignore diagonal elements with
   // fSig(i) < fTol * max(fSig)

   const Int_t    lwb       = fU.GetColLwb();
   const Int_t    upb       = lwb+fV.GetNcols()-1;
   const Double_t threshold = fSig(lwb)*fTol;

   TVectorD tmp(lwb,upb);
   for (Int_t irow = lwb; irow <= upb; irow++) {
      Double_t r = 0.0;
      if (fSig(irow) > threshold) {
         const TVectorD uc_i = TMatrixDColumn(fU,irow);
         r = uc_i * b;
         r /= fSig(irow);
      }
      tmp(irow) = r;
   }

   if (b.GetNrows() > fV.GetNrows()) {
      TVectorD tmp2;
      tmp2.Use(lwb,upb,b.GetMatrixArray());
      tmp2 = fV*tmp;
   } else
      b = fV*tmp;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompSVD::Solve(TMatrixDColumn &cb)
{
// Solve Ax=b assuming the SVD form of A is stored . Solution returned in the
// matrix column cb b.
// If A is of size (m x n), input vector b should be of size (m), however,
// the solution, returned in b, will be in the first (n) elements .
//
// For m > n , x  is the least-squares solution of min(A . x - b)

   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         return kFALSE;
      }
   }

   if (fU.GetNrows() != b->GetNrows() || fU.GetRowLwb() != b->GetRowLwb())
   {
      Error("Solve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   // We start with fU fSig fV^T x = b, and turn it into  fV^T x = fSig^-1 fU^T b
   // Form tmp = fSig^-1 fU^T b but ignore diagonal elements in
   // fSig(i) < fTol * max(fSig)

   const Int_t    lwb       = fU.GetColLwb();
   const Int_t    upb       = lwb+fV.GetNcols()-1;
   const Double_t threshold = fSig(lwb)*fTol;

   TVectorD tmp(lwb,upb);
   const TVectorD vb = cb;
   for (Int_t irow = lwb; irow <= upb; irow++) {
      Double_t r = 0.0;
      if (fSig(irow) > threshold) {
         const TVectorD uc_i = TMatrixDColumn(fU,irow);
         r = uc_i * vb;
         r /= fSig(irow);
      }
      tmp(irow) = r;
   }

   if (b->GetNrows() > fV.GetNrows()) {
      const TVectorD tmp2 = fV*tmp;
      TVectorD tmp3(cb);
      tmp3.SetSub(tmp2.GetLwb(),tmp2);
      cb = tmp3;
   } else
      cb = fV*tmp;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompSVD::TransSolve(TVectorD &b)
{
// Solve A^T x=b assuming the SVD form of A is stored . Solution returned in b.

   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         return kFALSE;
      }
   }

   if (fU.GetNrows() != fV.GetNrows() || fU.GetRowLwb() != fV.GetRowLwb()) {
      Error("TransSolve(TVectorD &","matrix should be square");
      return kFALSE;
   }

   if (fV.GetNrows() != b.GetNrows() || fV.GetRowLwb() != b.GetLwb())
   {
      Error("TransSolve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   // We start with fV fSig fU^T x = b, and turn it into  fU^T x = fSig^-1 fV^T b
   // Form tmp = fSig^-1 fV^T b but ignore diagonal elements in
   // fSig(i) < fTol * max(fSig)

   const Int_t    lwb       = fU.GetColLwb();
   const Int_t    upb       = lwb+fV.GetNcols()-1;
   const Double_t threshold = fSig(lwb)*fTol;

   TVectorD tmp(lwb,upb);
   for (Int_t i = lwb; i <= upb; i++) {
      Double_t r = 0.0;
      if (fSig(i) > threshold) {
         const TVectorD vc = TMatrixDColumn(fV,i);
         r = vc * b;
         r /= fSig(i);
      }
      tmp(i) = r;
   }
   b = fU*tmp;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDecompSVD::TransSolve(TMatrixDColumn &cb)
{
// Solve A^T x=b assuming the SVD form of A is stored . Solution returned in b.

   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         return kFALSE;
      }
   }

   if (fU.GetNrows() != fV.GetNrows() || fU.GetRowLwb() != fV.GetRowLwb()) {
      Error("TransSolve(TMatrixDColumn &","matrix should be square");
      return kFALSE;
   }

   if (fV.GetNrows() != b->GetNrows() || fV.GetRowLwb() != b->GetRowLwb())
   {
      Error("TransSolve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   // We start with fV fSig fU^T x = b, and turn it into  fU^T x = fSig^-1 fV^T b
   // Form tmp = fSig^-1 fV^T b but ignore diagonal elements in
   // fSig(i) < fTol * max(fSig)

   const Int_t    lwb       = fU.GetColLwb();
   const Int_t    upb       = lwb+fV.GetNcols()-1;
   const Double_t threshold = fSig(lwb)*fTol;

   const TVectorD vb = cb;
   TVectorD tmp(lwb,upb);
   for (Int_t i = lwb; i <= upb; i++) {
      Double_t r = 0.0;
      if (fSig(i) > threshold) {
         const TVectorD vc = TMatrixDColumn(fV,i);
         r = vc * vb;
         r /= fSig(i);
      }
      tmp(i) = r;
   }
   cb = fU*tmp;

   return kTRUE;
}

//______________________________________________________________________________
Double_t TDecompSVD::Condition()
{
// Matrix condition number

   if ( !TestBit(kCondition) ) {
      fCondition = -1;
      if (TestBit(kSingular))
         return fCondition;
      if ( !TestBit(kDecomposed) ) {
         if (!Decompose())
            return fCondition;
      }
      const Int_t colLwb = GetColLwb();
      const Int_t nCols  = GetNcols();
      const Double_t max = fSig(colLwb);
      const Double_t min = fSig(colLwb+nCols-1);
      fCondition = (min > 0.0) ? max/min : -1.0;
      SetBit(kCondition);
   }
   return fCondition;
}

//______________________________________________________________________________
void TDecompSVD::Det(Double_t &d1,Double_t &d2)
{
// Matrix determinant det = d1*TMath::Power(2.,d2)

   if ( !TestBit(kDetermined) ) {
      if ( !TestBit(kDecomposed) )
         Decompose();
      if (TestBit(kSingular)) {
         fDet1 = 0.0;
         fDet2 = 0.0;
      } else {
         DiagProd(fSig,fTol,fDet1,fDet2);
      }
      SetBit(kDetermined);
   }
   d1 = fDet1;
   d2 = fDet2;
}

//______________________________________________________________________________
Int_t  TDecompSVD::GetNrows  () const 
{ 
   return fU.GetNrows(); 
}

Int_t TDecompSVD::GetNcols  () const 
{
   return fV.GetNcols();
}

//______________________________________________________________________________
Bool_t TDecompSVD::Invert(TMatrixD &inv)
{
// For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
// The user should always supply a matrix of size (m x m) !
// If m > n , only the (n x m) part of the returned (pseudo inverse) matrix
// should be used .

   const Int_t rowLwb = GetRowLwb();
   const Int_t colLwb = GetColLwb();
   const Int_t nRows  = fU.GetNrows();

   if (inv.GetNrows()  != nRows  || inv.GetNcols()  != nRows ||
       inv.GetRowLwb() != rowLwb || inv.GetColLwb() != colLwb) {
      Error("Invert(TMatrixD &","Input matrix has wrong shape");
      return kFALSE;
   }

   inv.UnitMatrix();
   Bool_t status = MultiSolve(inv);

   return status;
}

//______________________________________________________________________________
TMatrixD TDecompSVD::Invert(Bool_t &status)
{
// For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
// (n x m) Ainv is returned .

   const Int_t rowLwb = GetRowLwb();
   const Int_t colLwb = GetColLwb();
   const Int_t rowUpb = rowLwb+fU.GetNrows()-1;
   TMatrixD inv(rowLwb,rowUpb,colLwb,colLwb+fU.GetNrows()-1);
   inv.UnitMatrix();
   status = MultiSolve(inv);
   inv.ResizeTo(rowLwb,rowLwb+fV.GetNcols()-1,colLwb,colLwb+fU.GetNrows()-1);

   return inv;
}

//______________________________________________________________________________
void TDecompSVD::Print(Option_t *opt) const
{
// Print class members

   TDecompBase::Print(opt);
   fU.Print("fU");
   fV.Print("fV");
   fSig.Print("fSig");
}

//______________________________________________________________________________
TDecompSVD &TDecompSVD::operator=(const TDecompSVD &source)
{
// Assignment operator

   if (this != &source) {
      TDecompBase::operator=(source);
      fU.ResizeTo(source.fU);
      fU = source.fU;
      fV.ResizeTo(source.fV);
      fV = source.fV;
      fSig.ResizeTo(source.fSig);
      fSig = source.fSig;
   }
   return *this;
}
